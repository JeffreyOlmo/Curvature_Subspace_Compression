import argparse
import json
import math
import os
import random
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from transformers import GPT2TokenizerFast

from ..curvature_regularizer import CurvatureRegularizer, attach_curvature_regularizer
from ..curvature_regularizer_shampoo import attach_shampoo_curvature_regularizer

URL_TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass
class ExperimentConfig:
    model_dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_mult: int = 4
    dropout: float = 0.1
    block_size: int = 128
    vocab_size: int = 50257  # Uses GPT-2 tokenizer
    batch_size: int = 32
    micro_batch: int = 1
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    device: str = "cuda"

    noise_fraction: float = 0.0  # fraction reserved for random noise (unused now)
    canary_count: int = 16
    canary_length: int = 20
    canary_repetitions: int = 50
    seed: int = 2025

    lambda_values: List[float] = None
    lambda_fc_mlp: Optional[float] = None
    ema_decay: float = 5e-4
    inv_update_interval: int = 25
    damping: float = 1e-2
    min_samples: int = 256
    track_stats: bool = True
    apply_to_mlp: bool = True
    apply_to_lm_head: bool = True
    max_train_tokens: Optional[int] = None
    max_val_tokens: Optional[int] = None
    use_shampoo: bool = False
    shampoo_preconditioning_steps: int = 25
    shampoo_matrix_eps: float = 1e-6
    lm_head_shard_size: int = 4096
    save_model_path: Optional[Path] = None
    resume_model_path: Optional[Path] = None
    dataset: str = "tiny_shakespeare"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_tiny_shakespeare(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "tiny_shakespeare.txt"
    if target.exists():
        return target
    with urllib.request.urlopen(URL_TINY_SHAKESPEARE) as response:
        text = response.read().decode("utf-8")
    with open(target, "w", encoding="utf-8") as f:
        f.write(text)
    return target


def chunk_tokens(tokens: torch.Tensor, block_size: int) -> torch.Tensor:
    total_tokens = tokens.size(0)
    usable = total_tokens // block_size * block_size
    tokens = tokens[:usable]
    chunks = tokens.view(-1, block_size)
    return chunks


class TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor):
        self.tokens = tokens

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.tokens[idx]
        x = data[:-1]
        y = data[1:]
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        dim = config.model_dim
        n_heads = config.n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, "model_dim must be divisible by n_heads"
        self.n_heads = n_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        dim = config.model_dim
        hidden = config.mlp_mult * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.model_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.model_dim)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.model_dim)
        self.pos_emb = nn.Embedding(config.block_size, config.model_dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block size."
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        tok = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def prepare_canaries(tokenizer: GPT2TokenizerFast, cfg: ExperimentConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    canaries = []
    for i in range(cfg.canary_count):
        identifier = rng.randrange(10**12, 10**13)
        canary = f"the secret sequence {identifier} ends here."
        canaries.append(canary)
    return canaries


def tokenize_text(tokenizer: GPT2TokenizerFast, text: str) -> torch.Tensor:
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.long)


def prepare_datasets(
    cfg: ExperimentConfig, cache_dir: Path
) -> Tuple[TokenDataset, TokenDataset, List[List[int]], GPT2TokenizerFast]:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset_key = cfg.dataset.lower().replace("-", "_")
    block_plus = cfg.block_size + 1

    def encode_texts(text_iterable: Iterable[str], target_tokens: Optional[int]) -> torch.Tensor:
        limit = target_tokens + block_plus if target_tokens is not None else None
        token_list: List[int] = []
        for text in text_iterable:
            if not text:
                continue
            encoded = tokenizer.encode(text)
            if limit is None:
                token_list.extend(encoded)
            else:
                remaining = limit - len(token_list)
                if remaining <= 0:
                    break
                if len(encoded) >= remaining:
                    token_list.extend(encoded[:remaining])
                    break
                token_list.extend(encoded)
        if not token_list:
            raise ValueError(f"No tokens were produced for dataset '{cfg.dataset}'.")
        tokens_tensor = torch.tensor(token_list, dtype=torch.long)
        if limit is not None and tokens_tensor.numel() < limit:
            repeats = (limit + tokens_tensor.numel() - 1) // tokens_tensor.numel()
            tokens_tensor = tokens_tensor.repeat((repeats,))[:limit]
        return tokens_tensor

    if dataset_key in {"tiny_shakespeare", "tiny"}:
        # tiny shakespeare dataset path
        raw_path = download_tiny_shakespeare(cache_dir)
        with open(raw_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        train_cut = int(len(tokens) * 0.9)
        train_tokens = tokens[:train_cut]
        val_tokens = tokens[train_cut:]
    elif dataset_key in {"wikitext103", "wikitext", "wikitext_103"}:
        dataset_train = load_dataset(
            "wikitext", "wikitext-103-v1", split="train", cache_dir=str(cache_dir)
        )
        dataset_val = load_dataset(
            "wikitext", "wikitext-103-v1", split="validation", cache_dir=str(cache_dir)
        )
        train_tokens = encode_texts(dataset_train["text"], cfg.max_train_tokens)
        val_limit = cfg.max_val_tokens if cfg.max_val_tokens is not None else 131072
        val_tokens = encode_texts(dataset_val["text"], val_limit)
    else:
        raise ValueError(f"Unsupported dataset '{cfg.dataset}'.")

    def tile_to_length(token_tensor: torch.Tensor, target: Optional[int]) -> torch.Tensor:
        if target is None:
            return token_tensor
        if target <= len(token_tensor):
            return token_tensor[:target]
        repeats = (target + len(token_tensor) - 1) // len(token_tensor)
        return token_tensor.repeat(repeats)[:target]

    train_tokens = tile_to_length(train_tokens, cfg.max_train_tokens)
    val_tokens = tile_to_length(val_tokens, cfg.max_val_tokens)

    canaries = prepare_canaries(tokenizer, cfg)
    encoded_canaries = [tokenizer.encode(canary) for canary in canaries]

    for seq in encoded_canaries:
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        for _ in range(cfg.canary_repetitions):
            train_tokens = torch.cat([train_tokens, seq_tensor])

    block_plus = cfg.block_size + 1
    train_chunks = chunk_tokens(train_tokens, block_plus)
    val_chunks = chunk_tokens(val_tokens, block_plus)

    train_dataset = TokenDataset(train_chunks)
    val_dataset = TokenDataset(val_chunks)

    return train_dataset, val_dataset, encoded_canaries, tokenizer


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def evaluate_perplexity(model: TinyTransformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def evaluate_canaries(
    model: TinyTransformer,
    tokenizer: GPT2TokenizerFast,
    canaries: List[List[int]],
    device: torch.device,
) -> List[Dict[str, float]]:
    model.eval()
    results = []
    with torch.no_grad():
        for seq in canaries:
            context = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(context, targets)
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            total_logprob = token_log_probs.sum().item()
            avg_logprob = token_log_probs.mean().item()
            perplexity = math.exp(-avg_logprob)
            results.append(
                {
                    "length": len(seq),
                    "total_logprob": total_logprob,
                    "avg_logprob": avg_logprob,
                    "perplexity": perplexity,
                }
            )
    return results


def attach_regularizers(
    model: TinyTransformer,
    cfg: ExperimentConfig,
    lambda_fc2: float,
    lambda_fc_mlp: Optional[float],
    device: torch.device,
) -> Dict[str, CurvatureRegularizer]:
    regs: Dict[str, CurvatureRegularizer] = {}
    if cfg.apply_to_lm_head and lambda_fc2 > 0:
        if cfg.use_shampoo:
            regs["lm_head"] = attach_shampoo_curvature_regularizer(
                module=model.lm_head,
                lambda_crt=lambda_fc2,
                ema_decay=cfg.ema_decay,
                preconditioning_compute_steps=cfg.shampoo_preconditioning_steps,
                damping=cfg.damping,
                min_samples=cfg.min_samples,
                matrix_eps=cfg.shampoo_matrix_eps,
                shard_size=cfg.lm_head_shard_size,
                track_stats=cfg.track_stats,
                device=device,
            )
        else:
            regs["lm_head"] = attach_curvature_regularizer(
                module=model.lm_head,
                lambda_crt=lambda_fc2,
                ema_decay=cfg.ema_decay,
                inv_update_interval=cfg.inv_update_interval,
                damping=cfg.damping,
                min_samples=cfg.min_samples,
                device=device,
                track_stats=cfg.track_stats,
                shard_size=cfg.lm_head_shard_size if cfg.lm_head_shard_size else None,
            )

    if cfg.apply_to_mlp and lambda_fc_mlp is not None and lambda_fc_mlp > 0:
        for idx, block in enumerate(model.blocks):
            if cfg.use_shampoo:
                regs[f"block{idx}.mlp.fc2"] = attach_shampoo_curvature_regularizer(
                    module=block.mlp.fc2,
                    lambda_crt=lambda_fc_mlp,
                    ema_decay=cfg.ema_decay,
                    preconditioning_compute_steps=cfg.shampoo_preconditioning_steps,
                    damping=cfg.damping,
                    min_samples=cfg.min_samples,
                    matrix_eps=cfg.shampoo_matrix_eps,
                    shard_size=None,
                    track_stats=cfg.track_stats,
                    device=device,
                )
            else:
                regs[f"block{idx}.mlp.fc2"] = attach_curvature_regularizer(
                    module=block.mlp.fc2,
                    lambda_crt=lambda_fc_mlp,
                    ema_decay=cfg.ema_decay,
                    inv_update_interval=cfg.inv_update_interval,
                    damping=cfg.damping,
                    min_samples=cfg.min_samples,
                    device=device,
                    track_stats=cfg.track_stats,
                )
    return regs


def train_one_model(
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    canaries: List[List[int]],
    tokenizer: GPT2TokenizerFast,
    device: torch.device,
    lambda_fc2: Optional[float],
    lambda_fc_mlp: Optional[float],
    seed: int,
) -> Tuple[Dict[str, object], TinyTransformer]:
    set_seed(seed)
    model = TinyTransformer(cfg).to(device)
    if cfg.resume_model_path is not None:
        state_dict = torch.load(cfg.resume_model_path, map_location=device)
        model.load_state_dict(state_dict)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))

    regularizers = attach_regularizers(model, cfg, lambda_fc2 or 0.0, lambda_fc_mlp, device)
    train_loss_history: List[float] = []
    train_perplexity_history: List[float] = []
    val_perplexity_history: List[float] = []
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            scheduler.step()

        avg_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_loss)
        train_perplexity_history.append(math.exp(avg_loss))

        model.eval()
        with torch.no_grad():
            val_ppl_epoch = evaluate_perplexity(model, val_loader, device)
        val_perplexity_history.append(val_ppl_epoch)

        if regularizers:
            stats_strings = []
            for name, reg in regularizers.items():
                stats = reg.collect_metrics()
                if stats:
                    stats_strings.append(f"{name}: avg_ratio={stats.get('avg_ratio', 0.0):.4f}")
            printable = "; ".join(stats_strings) if stats_strings else "no stats"
            print(f"[Epoch {epoch+1:02d}/{cfg.epochs}] loss={avg_loss:.4f} | CRT {printable}")
        else:
            print(f"[Epoch {epoch+1:02d}/{cfg.epochs}] loss={avg_loss:.4f}")

    train_ppl = evaluate_perplexity(model, train_loader, device)
    val_ppl = evaluate_perplexity(model, val_loader, device)
    canary_metrics = evaluate_canaries(model, tokenizer, canaries, device)

    result = {
        "lambda_fc2": lambda_fc2,
        "lambda_fc_mlp": lambda_fc_mlp,
        "train_perplexity": train_ppl,
        "val_perplexity": val_ppl,
        "canary_metrics": canary_metrics,
        "train_loss_history": train_loss_history,
        "train_perplexity_history": train_perplexity_history,
        "val_perplexity_history": val_perplexity_history,
    }

    for reg in regularizers.values():
        reg.close()

    model_cpu = model.to("cpu")
    torch.cuda.empty_cache()
    return result, model_cpu


def run_experiment(cfg: ExperimentConfig, cache_dir: Path, output: Optional[Path]) -> None:
    if cfg.lambda_values is None:
        cfg.lambda_values = [0.0, 3e-4, 1e-3]
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    train_dataset, val_dataset, canaries, tokenizer = prepare_datasets(cfg, cache_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    cfg_dict = asdict(cfg)
    if cfg_dict.get("save_model_path") is not None:
        cfg_dict["save_model_path"] = str(cfg_dict["save_model_path"])
    if cfg_dict.get("resume_model_path") is not None:
        cfg_dict["resume_model_path"] = str(cfg_dict["resume_model_path"])

    results: Dict[str, object] = {
        "config": cfg_dict,
        "canaries": canaries,
    }

    baseline, baseline_model = train_one_model(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        canaries=canaries,
        tokenizer=tokenizer,
        device=device,
        lambda_fc2=None,
        lambda_fc_mlp=None,
        seed=cfg.seed,
    )
    results["baseline"] = {
        "train_perplexity": baseline["train_perplexity"],
        "val_perplexity": baseline["val_perplexity"],
        "canary_metrics": baseline["canary_metrics"],
        "train_loss_history": baseline.get("train_loss_history", []),
        "train_perplexity_history": baseline.get("train_perplexity_history", []),
        "val_perplexity_history": baseline.get("val_perplexity_history", []),
    }

    run_summaries: List[Dict[str, object]] = []
    for idx, lambda_fc2 in enumerate(cfg.lambda_values):
        lambda_mlp = cfg.lambda_fc_mlp if cfg.lambda_fc_mlp is not None else lambda_fc2
        res, _ = train_one_model(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            canaries=canaries,
            tokenizer=tokenizer,
            device=device,
            lambda_fc2=lambda_fc2,
            lambda_fc_mlp=lambda_mlp,
            seed=cfg.seed + idx + 1,
        )
        run_summaries.append(
            {
                "lambda_crt_fc2": lambda_fc2,
                "lambda_crt_mlp": lambda_mlp,
                "train_perplexity": res["train_perplexity"],
                "val_perplexity": res["val_perplexity"],
                "canary_metrics": res["canary_metrics"],
                "delta_val_perplexity": res["val_perplexity"] - baseline["val_perplexity"],
                "train_loss_history": res.get("train_loss_history", []),
                "train_perplexity_history": res.get("train_perplexity_history", []),
                "val_perplexity_history": res.get("val_perplexity_history", []),
            }
        )

    results["runs"] = run_summaries

    if cfg.save_model_path is not None:
        cfg.save_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(baseline_model.state_dict(), cfg.save_model_path)
        print(f"Saved trained model to {cfg.save_model_path}")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output}")
    else:
        print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRT on tiny transformer with text canaries.")
    parser.add_argument("--cache-dir", type=Path, default=Path("./data"), help="Where to cache datasets/tokens.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-crt", type=float, nargs="+", default=[3e-4, 1e-3])
    parser.add_argument("--lambda-crt-mlp", type=float, default=None)
    parser.add_argument("--canary-count", type=int, default=16)
    parser.add_argument("--canary-repetitions", type=int, default=50)
    parser.add_argument("--apply-to-mlp", action="store_true")
    parser.add_argument("--apply-to-lm-head", action="store_true")
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-val-tokens", type=int, default=None)
    parser.add_argument("--use-shampoo-crt", action="store_true")
    parser.add_argument("--shampoo-preconditioning-steps", type=int, default=25)
    parser.add_argument("--shampoo-matrix-eps", type=float, default=1e-6)
    parser.add_argument("--lm-head-shard-size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-model", type=Path, default=None, help="Optional path to save the trained baseline model state dict.")
    parser.add_argument("--resume-model", type=Path, default=None, help="Optional checkpoint to resume training from.")
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny_shakespeare",
        help="Dataset to train on (e.g. tiny_shakespeare, wikitext103).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        canary_count=args.canary_count,
        canary_repetitions=args.canary_repetitions,
        lambda_values=args.lambda_crt,
        lambda_fc_mlp=args.lambda_crt_mlp,
        apply_to_mlp=args.apply_to_mlp,
        apply_to_lm_head=args.apply_to_lm_head or not args.apply_to_mlp,
        max_train_tokens=args.max_train_tokens,
        max_val_tokens=args.max_val_tokens,
        use_shampoo=args.use_shampoo_crt,
        shampoo_preconditioning_steps=args.shampoo_preconditioning_steps,
        shampoo_matrix_eps=args.shampoo_matrix_eps,
        lm_head_shard_size=args.lm_head_shard_size,
        device=args.device,
        seed=args.seed,
        save_model_path=args.save_model,
        resume_model_path=args.resume_model,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_mult=args.mlp_mult,
        block_size=args.block_size,
        dataset=args.dataset,
    )
    run_experiment(cfg=cfg, cache_dir=args.cache_dir, output=args.output)


if __name__ == "__main__":
    main()

