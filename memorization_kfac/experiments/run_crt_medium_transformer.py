import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import GPT2TokenizerFast

from ..curvature_regularizer import CurvatureRegularizer, attach_curvature_regularizer
from ..curvature_regularizer_shampoo import ShampooCurvatureRegularizer, attach_shampoo_curvature_regularizer


@dataclass
class ExperimentConfig:
    model_dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_mult: int = 4
    dropout: float = 0.1
    block_size: int = 256
    vocab_size: int = 50257
    batch_size: int = 64
    micro_batch: int = 64
    epochs: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    device: str = "cuda"

    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    train_split: str = "train"
    val_split: str = "validation"
    dataset_text_field: str = "text"
    max_train_tokens: int = 2_000_000
    max_val_tokens: int = 500_000
    streaming: bool = True

    canary_count: int = 32
    canary_length: int = 20
    canary_repetitions: int = 20
    seed: int = 2025

    lambda_values: List[float] = None
    lambda_fc_mlp: Optional[float] = None
    lambda_fc_attn: Optional[float] = None
    ema_decay: float = 5e-4
    inv_update_interval: int = 25
    damping: float = 1e-2
    min_samples: int = 256
    track_stats: bool = True
    apply_to_mlp: bool = True
    apply_to_attn: bool = False
    apply_to_lm_head: bool = True
    lm_head_shard_size: int = 4096
    use_shampoo: bool = False
    shampoo_preconditioning_steps: int = 4
    shampoo_matrix_eps: float = 1e-6
    freeze_lm_head: bool = False


def setup_distributed(preferred_device: str) -> Tuple[torch.device, bool, int, int, int]:
    if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return device, True, rank, world_size, local_rank

    device = torch.device(
        preferred_device if torch.cuda.is_available() and preferred_device.startswith("cuda") else "cpu"
    )
    return device, False, 0, 1, 0


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TokenBlockDataset(Dataset):
    def __init__(self, blocks: torch.Tensor):
        self.blocks = blocks

    def __len__(self) -> int:
        return self.blocks.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.blocks[idx]
        return data[:-1], data[1:]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        dim = config.model_dim
        n_heads = config.n_heads
        assert dim % n_heads == 0, "model_dim must be divisible by n_heads"
        self.head_dim = dim // n_heads
        self.n_heads = n_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(dim, dim=2)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = self.mask[:, :, :seq_len, :seq_len]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
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


class TransformerBlock(nn.Module):
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


class MediumTransformer(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.model_dim)
        self.pos_emb = nn.Embedding(config.block_size, config.model_dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block size.")

        pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
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
    for _ in range(cfg.canary_count):
        identifier = rng.randrange(10**12, 10**13)
        canary = f"disclose private key {identifier} end transmission."
        canaries.append(canary)
    return canaries


def iter_dataset_texts(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_field: str,
    streaming: bool,
) -> Iterable[str]:
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=streaming,
    )
    for sample in dataset:
        text = sample.get(text_field, "")
        if not text:
            continue
        if not text.endswith("\n"):
            text = text + "\n"
        yield text


def gather_tokens(
    tokenizer: GPT2TokenizerFast,
    texts: Iterable[str],
    max_tokens: int,
) -> torch.Tensor:
    tokens: List[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        tokens.extend(ids)
        if len(tokens) >= max_tokens:
            break
    if len(tokens) < max_tokens:
        max_tokens = len(tokens)
    if max_tokens == 0:
        raise RuntimeError("No tokens collected from dataset.")
    tensor = torch.tensor(tokens[:max_tokens], dtype=torch.long)
    return tensor


def chunk_tokens(tokens: torch.Tensor, block_size: int) -> torch.Tensor:
    block_plus = block_size + 1
    usable = tokens.size(0) // block_plus * block_plus
    tokens = tokens[:usable]
    if usable == 0:
        raise RuntimeError("Not enough tokens to form a single block.")
    return tokens.view(-1, block_plus)


def prepare_datasets(cfg: ExperimentConfig, cache_dir: Path) -> Tuple[TokenBlockDataset, TokenBlockDataset, List[List[int]], GPT2TokenizerFast]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_texts = iter_dataset_texts(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.train_split,
        text_field=cfg.dataset_text_field,
        streaming=cfg.streaming,
    )
    val_texts = iter_dataset_texts(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.val_split,
        text_field=cfg.dataset_text_field,
        streaming=cfg.streaming,
    )

    train_tokens = gather_tokens(tokenizer, train_texts, cfg.max_train_tokens)
    val_tokens = gather_tokens(tokenizer, val_texts, cfg.max_val_tokens)

    canaries = prepare_canaries(tokenizer, cfg)
    encoded_canaries = [tokenizer.encode(canary, add_special_tokens=False) for canary in canaries]
    extra_tokens: List[torch.Tensor] = []
    for seq in encoded_canaries:
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        extra_tokens.append(seq_tensor.repeat(cfg.canary_repetitions))
    if extra_tokens:
        train_tokens = torch.cat([train_tokens, *extra_tokens])

    train_blocks = chunk_tokens(train_tokens, cfg.block_size)
    val_blocks = chunk_tokens(val_tokens, cfg.block_size)

    train_dataset = TokenBlockDataset(train_blocks)
    val_dataset = TokenBlockDataset(val_blocks)

    return train_dataset, val_dataset, encoded_canaries, tokenizer


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def evaluate_perplexity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: bool,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, loss = model(x, y)
            total_loss += float(loss.item()) * y.numel()
            total_tokens += float(y.numel())

    stats = torch.tensor([total_loss, total_tokens], dtype=torch.float64, device=device)
    if distributed and dist.is_initialized():
        dist.all_reduce(stats)
    total_loss, total_tokens = stats.tolist()
    avg_loss = total_loss / max(1.0, total_tokens)
    if was_training:
        model.train()
    return math.exp(avg_loss)


def evaluate_canaries(
    model: nn.Module,
    tokenizer: GPT2TokenizerFast,
    canaries: List[List[int]],
    device: torch.device,
) -> List[Dict[str, float]]:
    was_training = model.training
    model.eval()
    metrics: List[Dict[str, float]] = []
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
            metrics.append(
                {
                    "length": len(seq),
                    "total_logprob": total_logprob,
                    "avg_logprob": avg_logprob,
                    "perplexity": perplexity,
                }
            )
    if was_training:
        model.train()
    return metrics


def attach_regularizers(
    model: MediumTransformer,
    cfg: ExperimentConfig,
    lambda_fc2: float,
    lambda_fc_mlp: Optional[float],
    lambda_fc_attn: Optional[float],
    device: torch.device,
) -> Dict[str, CurvatureRegularizer]:
    regs: Dict[str, CurvatureRegularizer] = {}
    if not cfg.apply_to_attn:
        lambda_attn = None
    else:
        lambda_attn = (
            lambda_fc_attn
            if lambda_fc_attn is not None
            else (lambda_fc_mlp if lambda_fc_mlp is not None else lambda_fc2)
        )
    if cfg.use_shampoo:
        if cfg.apply_to_lm_head and lambda_fc2 > 0:
            if cfg.freeze_lm_head:
                raise ValueError("Cannot apply CRT to LM head while it is frozen.")
            regs["lm_head"] = attach_shampoo_curvature_regularizer(
                module=model.lm_head,
                lambda_crt=lambda_fc2,
                ema_decay=cfg.ema_decay,
                matrix_eps=cfg.shampoo_matrix_eps,
                preconditioning_compute_steps=cfg.shampoo_preconditioning_steps,
                damping=cfg.damping,
                min_samples=cfg.min_samples,
                shard_size=cfg.lm_head_shard_size,
                track_stats=cfg.track_stats,
                device=device,
            )
        if cfg.apply_to_attn and lambda_attn is not None and lambda_attn > 0:
            for idx, block in enumerate(model.blocks):
                regs[f"block{idx}.attn.qkv"] = attach_shampoo_curvature_regularizer(
                    module=block.attn.qkv,
                    lambda_crt=lambda_attn,
                    ema_decay=cfg.ema_decay,
                    matrix_eps=cfg.shampoo_matrix_eps,
                    preconditioning_compute_steps=cfg.shampoo_preconditioning_steps,
                    damping=cfg.damping,
                    min_samples=cfg.min_samples,
                    shard_size=None,
                    track_stats=cfg.track_stats,
                    device=device,
                )
                regs[f"block{idx}.attn.proj"] = attach_shampoo_curvature_regularizer(
                    module=block.attn.proj,
                    lambda_crt=lambda_attn,
                    ema_decay=cfg.ema_decay,
                    matrix_eps=cfg.shampoo_matrix_eps,
                    preconditioning_compute_steps=cfg.shampoo_preconditioning_steps,
                    damping=cfg.damping,
                    min_samples=cfg.min_samples,
                    shard_size=None,
                    track_stats=cfg.track_stats,
                    device=device,
                )
        if cfg.apply_to_mlp and lambda_fc_mlp is not None and lambda_fc_mlp > 0:
            for idx, block in enumerate(model.blocks):
                regs[f"block{idx}.mlp.fc1"] = attach_shampoo_curvature_regularizer(
                    module=block.mlp.fc1,
                    lambda_crt=lambda_fc_mlp,
                    ema_decay=cfg.ema_decay,
                    matrix_eps=cfg.shampoo_matrix_eps,
                    preconditioning_compute_steps=cfg.shampoo_preconditioning_steps,
                    damping=cfg.damping,
                    min_samples=cfg.min_samples,
                    shard_size=None,
                    track_stats=cfg.track_stats,
                    device=device,
                )
                regs[f"block{idx}.mlp.fc2"] = attach_shampoo_curvature_regularizer(
                    module=block.mlp.fc2,
                    lambda_crt=lambda_fc_mlp,
                    ema_decay=cfg.ema_decay,
                    matrix_eps=cfg.shampoo_matrix_eps,
                    preconditioning_compute_steps=cfg.shampoo_preconditioning_steps,
                    damping=cfg.damping,
                    min_samples=cfg.min_samples,
                    shard_size=None,
                    track_stats=cfg.track_stats,
                    device=device,
                )
        return regs

    if cfg.apply_to_lm_head and lambda_fc2 > 0:
        if cfg.freeze_lm_head:
            raise ValueError("Cannot apply CRT to LM head while it is frozen.")
        regs["lm_head"] = attach_curvature_regularizer(
            module=model.lm_head,
            lambda_crt=lambda_fc2,
            ema_decay=cfg.ema_decay,
            inv_update_interval=cfg.inv_update_interval,
            damping=cfg.damping,
            min_samples=cfg.min_samples,
            device=device,
            track_stats=cfg.track_stats,
            shard_size=cfg.lm_head_shard_size,
        )
    if cfg.apply_to_attn and lambda_attn is not None and lambda_attn > 0:
        for idx, block in enumerate(model.blocks):
            regs[f"block{idx}.attn.qkv"] = attach_curvature_regularizer(
                module=block.attn.qkv,
                lambda_crt=lambda_attn,
                ema_decay=cfg.ema_decay,
                inv_update_interval=cfg.inv_update_interval,
                damping=cfg.damping,
                min_samples=cfg.min_samples,
                device=device,
                track_stats=cfg.track_stats,
            )
            regs[f"block{idx}.attn.proj"] = attach_curvature_regularizer(
                module=block.attn.proj,
                lambda_crt=lambda_attn,
                ema_decay=cfg.ema_decay,
                inv_update_interval=cfg.inv_update_interval,
                damping=cfg.damping,
                min_samples=cfg.min_samples,
                device=device,
                track_stats=cfg.track_stats,
            )
    if cfg.apply_to_mlp and lambda_fc_mlp is not None and lambda_fc_mlp > 0:
        for idx, block in enumerate(model.blocks):
            regs[f"block{idx}.mlp.fc1"] = attach_curvature_regularizer(
                module=block.mlp.fc1,
                lambda_crt=lambda_fc_mlp,
                ema_decay=cfg.ema_decay,
                inv_update_interval=cfg.inv_update_interval,
                damping=cfg.damping,
                min_samples=cfg.min_samples,
                device=device,
                track_stats=cfg.track_stats,
            )
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
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    canaries: List[List[int]],
    tokenizer: GPT2TokenizerFast,
    device: torch.device,
    lambda_fc2: Optional[float],
    lambda_fc_mlp: Optional[float],
    lambda_fc_attn: Optional[float],
    seed: int,
    distributed: bool,
    world_size: int,
    rank: int,
) -> Dict[str, object]:
    set_seed(seed + rank)
    model = MediumTransformer(cfg).to(device)
    if cfg.freeze_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = False

    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    grad_accum_steps = max(1, math.ceil(cfg.batch_size / (cfg.micro_batch * world_size)))
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = cfg.epochs * max(1, steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    regularizers = attach_regularizers(
        model,
        cfg,
        lambda_fc2 or 0.0,
        lambda_fc_mlp,
        lambda_fc_attn,
        device,
    )
    ddp_model: nn.Module = model
    if distributed:
        device_ids = [device.index] if device.type == "cuda" else None
        ddp_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    is_main = (rank == 0)
    pin_memory = device.type == "cuda"

    for epoch in range(cfg.epochs):
        ddp_model.train()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)
            _, loss = ddp_model(x, y)
            loss_value = loss.item()
            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
            running_loss += loss_value

        avg_loss = running_loss / max(1, len(train_loader))
        if regularizers and is_main:
            if cfg.track_stats:
                stats_strings = []
                for name, reg in regularizers.items():
                    stats = reg.collect_metrics()
                    if stats:
                        stats_strings.append(f"{name}: avg_ratio={stats.get('avg_ratio', 0.0):.4f}")
                printable = "; ".join(stats_strings) if stats_strings else "no stats"
                print(f"[Epoch {epoch+1:02d}/{cfg.epochs}] loss={avg_loss:.4f} | CRT {printable}")
            else:
                print(f"[Epoch {epoch+1:02d}/{cfg.epochs}] loss={avg_loss:.4f}")
        elif is_main:
            print(f"[Epoch {epoch+1:02d}/{cfg.epochs}] loss={avg_loss:.4f}")

    if distributed and isinstance(train_eval_loader.sampler, DistributedSampler):
        train_eval_loader.sampler.set_epoch(0)
    if distributed and isinstance(val_loader.sampler, DistributedSampler):
        val_loader.sampler.set_epoch(0)

    train_ppl = evaluate_perplexity(ddp_model, train_eval_loader, device, distributed)
    val_ppl = evaluate_perplexity(ddp_model, val_loader, device, distributed)
    canary_metrics: List[Dict[str, float]] = []
    if is_main:
        canary_metrics = evaluate_canaries(ddp_model, tokenizer, canaries, device)
    if distributed and dist.is_initialized():
        obj_list: List[List[Dict[str, float]]] = [canary_metrics]
        dist.broadcast_object_list(obj_list, src=0)
        canary_metrics = obj_list[0]

    result = {
        "lambda_fc2": lambda_fc2,
        "lambda_fc_mlp": lambda_fc_mlp,
        "lambda_fc_attn": lambda_fc_attn,
        "train_perplexity": train_ppl,
        "val_perplexity": val_ppl,
        "canary_metrics": canary_metrics,
    }

    for reg in regularizers.values():
        reg.close()

    if distributed and dist.is_initialized():
        obj_list_result = [result]
        dist.broadcast_object_list(obj_list_result, src=0)
        result = obj_list_result[0]

    return result


def run_experiment(
    cfg: ExperimentConfig,
    cache_dir: Path,
    output: Optional[Path],
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
) -> None:
    if cfg.lambda_values is None:
        cfg.lambda_values = [0.0, 3e-4, 1e-3]
    if cfg.micro_batch <= 0 or cfg.batch_size <= 0:
        raise ValueError("Batch size and micro batch must be positive integers.")
    is_main = rank == 0
    if is_main:
        print(f"Using device: {device}, distributed={distributed}, world_size={world_size}")

    train_dataset, val_dataset, canaries, tokenizer = prepare_datasets(cfg, cache_dir)

    train_sampler: Optional[DistributedSampler] = None
    train_eval_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        train_eval_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    pin_memory = device.type == "cuda"
    num_workers = 2 if pin_memory else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.micro_batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_batch,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=cfg.micro_batch,
        shuffle=False,
        sampler=train_eval_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.micro_batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_batch,
    )

    results: Dict[str, object] = {
        "config": asdict(cfg),
        "train_blocks": len(train_dataset),
        "val_blocks": len(val_dataset),
        "canaries": canaries,
    }

    baseline = train_one_model(
        cfg=cfg,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        canaries=canaries,
        tokenizer=tokenizer,
        device=device,
        lambda_fc2=None,
        lambda_fc_mlp=None,
        lambda_fc_attn=None,
        seed=cfg.seed,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
    )
    results["baseline"] = {
        "train_perplexity": baseline["train_perplexity"],
        "val_perplexity": baseline["val_perplexity"],
        "canary_metrics": baseline["canary_metrics"],
    }

    run_summaries: List[Dict[str, object]] = []
    for idx, lambda_fc2 in enumerate(cfg.lambda_values):
        lambda_mlp = cfg.lambda_fc_mlp if cfg.lambda_fc_mlp is not None else lambda_fc2
        lambda_attn = (
            cfg.lambda_fc_attn
            if cfg.lambda_fc_attn is not None
            else (lambda_mlp if lambda_mlp is not None else lambda_fc2)
        )
        res = train_one_model(
            cfg=cfg,
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            canaries=canaries,
            tokenizer=tokenizer,
            device=device,
            lambda_fc2=lambda_fc2,
            lambda_fc_mlp=lambda_mlp,
            lambda_fc_attn=lambda_attn,
            seed=cfg.seed + idx + 1,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
        )
        run_summaries.append(
            {
                "lambda_crt_fc2": lambda_fc2,
                "lambda_crt_mlp": lambda_mlp,
                "lambda_crt_attn": lambda_attn,
                "train_perplexity": res["train_perplexity"],
                "val_perplexity": res["val_perplexity"],
                "canary_metrics": res["canary_metrics"],
                "delta_val_perplexity": res["val_perplexity"] - baseline["val_perplexity"],
            }
        )

    results["runs"] = run_summaries

    if distributed and dist.is_initialized():
        dist.barrier()

    if is_main:
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {output}")
        else:
            print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRT on medium transformer with larger dataset.")
    parser.add_argument("--cache-dir", type=Path, default=Path("./data"), help="Where to cache datasets/tokens.")
    parser.add_argument("--dataset-name", type=str, default="wikitext", help="HuggingFace dataset name.")
    parser.add_argument(
        "--dataset-config", type=str, default="wikitext-103-raw-v1", help="Dataset configuration name if required."
    )
    parser.add_argument("--train-split", type=str, default="train", help="Train split name.")
    parser.add_argument("--val-split", type=str, default="validation", help="Validation split name.")
    parser.add_argument("--dataset-text-field", type=str, default="text", help="Field containing text.")
    parser.add_argument("--max-train-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-val-tokens", type=int, default=500_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64, help="Effective batch size (after accumulation).")
    parser.add_argument("--micro-batch", type=int, default=64, help="Per-step batch size.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-crt", type=float, nargs="+", default=[3e-4, 1e-3])
    parser.add_argument("--lambda-crt-mlp", type=float, default=None)
    parser.add_argument("--lambda-crt-attn", type=float, default=None)
    parser.add_argument("--canary-count", type=int, default=32)
    parser.add_argument("--canary-repetitions", type=int, default=20)
    parser.add_argument("--apply-to-mlp", action="store_true", help="Force-enable CRT on MLP blocks.")
    parser.add_argument("--disable-mlp", action="store_true", help="Disable CRT on MLP blocks.")
    parser.add_argument("--apply-to-attn", action="store_true", help="Force-enable CRT on attention projections.")
    parser.add_argument("--disable-attn", action="store_true", help="Disable CRT on attention projections.")
    parser.add_argument("--apply-to-lm-head", action="store_true", help="Force-enable CRT on LM head.")
    parser.add_argument("--disable-lm-head", action="store_true", help="Disable CRT on LM head.")
    parser.add_argument("--freeze-lm-head", action="store_true", help="Freeze LM head weights (no optimizer updates).")
    parser.add_argument("--use-shampoo-crt", action="store_true", help="Use Shampoo-based curvature regularizer.")
    parser.add_argument("--shampoo-preconditioning-steps", type=int, default=4, help="Steps between Shampoo updates.")
    parser.add_argument("--shampoo-matrix-eps", type=float, default=1e-6, help="Matrix epsilon for Shampoo.")
    parser.add_argument(
        "--lm-head-shard-size",
        type=int,
        default=4096,
        help="Shard size to use when computing LM head curvature (rows per shard).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--no-streaming", action="store_false", dest="streaming", help="Disable HF streaming mode.")
    parser.set_defaults(streaming=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, distributed, rank, world_size, _local_rank = setup_distributed(args.device)
    default_cfg = ExperimentConfig()
    apply_to_mlp = default_cfg.apply_to_mlp
    if args.apply_to_mlp:
        apply_to_mlp = True
    if args.disable_mlp:
        apply_to_mlp = False

    apply_to_attn = default_cfg.apply_to_attn
    if args.apply_to_attn:
        apply_to_attn = True
    if args.disable_attn:
        apply_to_attn = False

    apply_to_lm_head = default_cfg.apply_to_lm_head
    if args.apply_to_lm_head:
        apply_to_lm_head = True
    if args.disable_lm_head:
        apply_to_lm_head = False

    freeze_lm_head = default_cfg.freeze_lm_head
    if args.freeze_lm_head:
        freeze_lm_head = True
        apply_to_lm_head = False

    streaming = args.streaming

    cfg = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch=args.micro_batch,
        learning_rate=args.lr,
        canary_count=args.canary_count,
        canary_repetitions=args.canary_repetitions,
        lambda_values=args.lambda_crt,
        lambda_fc_mlp=args.lambda_crt_mlp,
        lambda_fc_attn=args.lambda_crt_attn,
        apply_to_mlp=apply_to_mlp,
        apply_to_attn=apply_to_attn,
        apply_to_lm_head=apply_to_lm_head,
        lm_head_shard_size=args.lm_head_shard_size,
        use_shampoo=args.use_shampoo_crt,
        shampoo_preconditioning_steps=args.shampoo_preconditioning_steps,
        shampoo_matrix_eps=args.shampoo_matrix_eps,
        freeze_lm_head=freeze_lm_head,
        seed=args.seed,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        val_split=args.val_split,
        dataset_text_field=args.dataset_text_field,
        max_train_tokens=args.max_train_tokens,
        max_val_tokens=args.max_val_tokens,
        streaming=streaming,
        device=str(device),
    )
    try:
        run_experiment(
            cfg=cfg,
            cache_dir=args.cache_dir,
            output=args.output,
            device=device,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

