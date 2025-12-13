import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import Conv1D

from .run_crt_tiny_transformer import (
    ExperimentConfig,
    TinyTransformer,
    collate_batch,
    evaluate_canaries,
    evaluate_perplexity,
    prepare_datasets,
    set_seed,
)
from ..subspace_linear import CurvatureSubspaceLinear


def _fraction_to_key(fraction: float) -> Tuple[str, str]:
    if math.isclose(fraction, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        return "baseline", "Baseline"
    pct = int(round(fraction * 100))
    return f"comp{pct:03d}", f"Keep {pct}%"


@dataclass
class SubspaceExperimentConfig(ExperimentConfig):
    """Configuration for curvature subspace compression on the tiny transformer."""

    cache_dir: Path = Path("./data")
    pretrained_model: Optional[str] = None
    checkpoint_path: Optional[Path] = None
    pretrain_epochs: int = 1
    finetune_epochs: int = 0
    variance_keep_in: float = 0.9
    variance_keep_out: float = 0.9
    curvature_ema_decay: float = 5e-4
    curvature_min_samples: int = 512
    curvature_damping: float = 1e-3
    finetune_lr: float = 1e-4
    finetune_weight_decay: float = 1e-2
    compress_fc1: bool = True
    compress_fc2: bool = True
    keep_fraction_in: Optional[float] = None
    keep_fraction_out: Optional[float] = None
    keep_fraction_sweep: Optional[Sequence[float]] = None
    output: Optional[Path] = None
    output_dir: Optional[Path] = None
    pretrain_apply_updates: bool = True


class GPT2Wrapper(nn.Module):
    """Adapter that aligns Hugging Face GPT-2 forward signature with TinyTransformer."""

    def __init__(self, model: GPT2LMHeadModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        outputs = self.model(input_ids=idx, labels=targets, use_cache=False)
        logits = outputs.logits
        loss = outputs.loss if targets is not None else None
        return logits, loss


def convert_conv1d_to_linear(module: Conv1D) -> nn.Linear:
    """Convert GPT-2 Conv1D modules (linear layers with transposed weights) into nn.Linear."""
    linear = nn.Linear(module.weight.size(0), module.weight.size(1), bias=True)
    linear = linear.to(device=module.weight.device, dtype=module.weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(module.weight.t().contiguous())
        linear.bias.copy_(module.bias)
    return linear


def get_submodule(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def set_submodule(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    module: nn.Module = root
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    last = parts[-1]
    if last.isdigit():
        module[int(last)] = new_module  # type: ignore[index]
    else:
        setattr(module, last, new_module)


def collect_target_paths(model: nn.Module, cfg: SubspaceExperimentConfig, use_pretrained: bool) -> List[str]:
    paths: List[str] = []
    if use_pretrained:
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
            raise ValueError("Expected GPT-2 style model with transformer.h blocks.")
        num_layers = len(model.transformer.h)  # type: ignore[attr-defined]
        for idx in range(num_layers):
            if cfg.compress_fc1:
                paths.append(f"transformer.h.{idx}.mlp.c_fc")
            if cfg.compress_fc2:
                paths.append(f"transformer.h.{idx}.mlp.c_proj")
    else:
        if not hasattr(model, "blocks"):
            raise ValueError("Expected TinyTransformer with attribute 'blocks'.")
        num_layers = len(model.blocks)  # type: ignore[attr-defined]
        for idx in range(num_layers):
            if cfg.compress_fc1:
                paths.append(f"blocks.{idx}.mlp.fc1")
            if cfg.compress_fc2:
                paths.append(f"blocks.{idx}.mlp.fc2")
    return paths


def _compute_rank_for_variance(evals: torch.Tensor, target: float) -> Tuple[int, float]:
    if evals.numel() == 0:
        return 0, 0.0

    clamped_target = max(0.0, min(float(target), 1.0))
    total = torch.clamp(evals.sum(), min=1e-12)

    if clamped_target >= 1.0:
        return evals.numel(), 1.0

    cumulative = torch.cumsum(evals, dim=0)
    required = clamped_target * total
    idx = torch.searchsorted(cumulative, required, right=False)
    rank = int(idx.item() + 1)
    rank = max(1, min(rank, evals.numel()))
    actual = (cumulative[rank - 1] / total).item()
    return rank, actual


class CurvatureAccumulator:
    """Tracks activation/gradient covariances for a linear layer during training."""

    def __init__(
        self,
        name: str,
        module: nn.Linear,
        device: torch.device,
        ema_decay: float,
        min_samples: int,
        damping: float,
    ) -> None:
        if not isinstance(module, nn.Linear):
            raise TypeError(f"CurvatureAccumulator expects nn.Linear, got {type(module)}")

        self.name = name
        self.module = module
        self.device = device
        self.ema_decay = ema_decay
        self.min_samples = min_samples
        self.damping = damping

        out_dim, in_dim = module.weight.shape
        self.A = torch.zeros(in_dim, in_dim, device=device, dtype=torch.float32)
        self.G = torch.zeros(out_dim, out_dim, device=device, dtype=torch.float32)
        self.samples_seen = 0
        self._tracking_started = False
        self._last_activation: Optional[torch.Tensor] = None

        self._forward_handle = module.register_forward_pre_hook(self._forward_hook, with_kwargs=False)
        self._backward_handle = module.register_full_backward_hook(self._backward_hook, prepend=False)

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def _forward_hook(self, module: nn.Module, inputs) -> None:
        if not inputs:
            self._last_activation = None
            return
        act = inputs[0]
        if act is None:
            self._last_activation = None
            return
        if act.dim() > 2:
            act = act.reshape(-1, act.size(-1))
        self._last_activation = act.detach().to(self.device, dtype=torch.float32)

    def _backward_hook(self, module: nn.Module, grad_input, grad_output) -> None:
        if not grad_output:
            self._last_activation = None
            return
        grad_out = grad_output[0]
        if grad_out is None:
            self._last_activation = None
            return
        grad_out = grad_out.detach().to(self.device, dtype=torch.float32)
        if grad_out.dim() > 2:
            grad_out = grad_out.reshape(-1, grad_out.size(-1))
        act = self._last_activation
        if act is None or act.size(0) == 0:
            self._last_activation = None
            return
        self._update_covariances(act, grad_out)
        self._last_activation = None

    def _update_covariances(self, act: torch.Tensor, grad_out: torch.Tensor) -> None:
        rho = self.ema_decay
        batch = act.size(0)
        norm = 1.0 / float(batch)
        A_batch = (act.t() @ act) * norm
        G_batch = (grad_out.t() @ grad_out) * norm

        if not self._tracking_started:
            self.A.copy_(A_batch)
            self.G.copy_(G_batch)
            self._tracking_started = True
        else:
            self.A.mul_(1.0 - rho).add_(A_batch, alpha=rho)
            self.G.mul_(1.0 - rho).add_(G_batch, alpha=rho)
        self.samples_seen += batch

    def _compute_bases(
        self,
        variance_keep_in: float,
        variance_keep_out: float,
        keep_fraction_in: Optional[float],
        keep_fraction_out: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if not self._tracking_started:
            raise RuntimeError(f"Curvature stats were never collected for layer {self.name}.")
        if self.samples_seen < self.min_samples:
            raise RuntimeError(
                f"Insufficient samples for layer {self.name}: "
                f"collected {self.samples_seen}, require {self.min_samples}."
            )

        device = self.module.weight.device
        dtype = self.module.weight.dtype
        in_dim = self.A.size(0)
        out_dim = self.G.size(0)

        A_sym = 0.5 * (self.A + self.A.t())
        G_sym = 0.5 * (self.G + self.G.t())

        eye_in = torch.eye(in_dim, device=self.device, dtype=self.A.dtype)
        eye_out = torch.eye(out_dim, device=self.device, dtype=self.G.dtype)
        A_damped = A_sym + self.damping * eye_in
        G_damped = G_sym + self.damping * eye_out

        A_cpu = A_damped.detach().to(dtype=torch.float64, device="cpu")
        G_cpu = G_damped.detach().to(dtype=torch.float64, device="cpu")

        eigvals_in, eigvecs_in = torch.linalg.eigh(A_cpu)
        eigvals_out, eigvecs_out = torch.linalg.eigh(G_cpu)

        eigvals_in = torch.clamp(eigvals_in, min=1e-12)
        eigvals_out = torch.clamp(eigvals_out, min=1e-12)

        idx_in = torch.argsort(eigvals_in, descending=True)
        idx_out = torch.argsort(eigvals_out, descending=True)

        eigvals_in = eigvals_in[idx_in]
        eigvecs_in = eigvecs_in[:, idx_in]
        eigvals_out = eigvals_out[idx_out]
        eigvecs_out = eigvecs_out[:, idx_out]

        if keep_fraction_in is not None:
            frac_in = float(max(0.0, min(keep_fraction_in, 1.0)))
            rank_in = max(1, int(math.ceil(frac_in * in_dim)))
            actual_in = float((torch.cumsum(eigvals_in, dim=0)[rank_in - 1] / eigvals_in.sum()).item())
            keep_in_actual_fraction = rank_in / in_dim
        else:
            rank_in, actual_in = _compute_rank_for_variance(eigvals_in, variance_keep_in)
            keep_in_actual_fraction = rank_in / in_dim

        if keep_fraction_out is not None:
            frac_out = float(max(0.0, min(keep_fraction_out, 1.0)))
            rank_out = max(1, int(math.ceil(frac_out * out_dim)))
            actual_out = float((torch.cumsum(eigvals_out, dim=0)[rank_out - 1] / eigvals_out.sum()).item())
            keep_out_actual_fraction = rank_out / out_dim
        else:
            rank_out, actual_out = _compute_rank_for_variance(eigvals_out, variance_keep_out)
            keep_out_actual_fraction = rank_out / out_dim

        left_cpu = eigvecs_out[:, :rank_out].contiguous()
        right_cpu = eigvecs_in[:, :rank_in].contiguous()

        left = left_cpu.to(device=device, dtype=dtype)
        right = right_cpu.to(device=device, dtype=dtype)

        info = {
            "layer": self.name,
            "rank_in": rank_in,
            "rank_out": rank_out,
            "variance_keep_in_target": float(variance_keep_in),
            "variance_keep_out_target": float(variance_keep_out),
            "variance_keep_in_actual": actual_in,
            "variance_keep_out_actual": actual_out,
            "samples_seen": int(self.samples_seen),
            "min_samples": int(self.min_samples),
            "keep_fraction_in_target": float(keep_fraction_in) if keep_fraction_in is not None else None,
            "keep_fraction_out_target": float(keep_fraction_out) if keep_fraction_out is not None else None,
            "keep_fraction_in_actual": keep_in_actual_fraction,
            "keep_fraction_out_actual": keep_out_actual_fraction,
        }
        return left, right, info

    def build_compressed_layer(
        self,
        variance_keep_in: float,
        variance_keep_out: float,
        keep_fraction_in: Optional[float],
        keep_fraction_out: Optional[float],
    ) -> Tuple[CurvatureSubspaceLinear, Dict[str, float]]:
        left_basis, right_basis, info = self._compute_bases(
            variance_keep_in,
            variance_keep_out,
            keep_fraction_in,
            keep_fraction_out,
        )
        compressed = CurvatureSubspaceLinear.from_linear(self.module, left_basis, right_basis)

        with torch.no_grad():
            original_weight = self.module.weight.detach().to(dtype=torch.float64, device="cpu")
            compressed_weight = compressed.materialize_weight().detach().to(dtype=torch.float64, device="cpu")
            weight_norm = torch.linalg.vector_norm(original_weight)
            if weight_norm.item() > 0:
                diff_norm = torch.linalg.vector_norm(original_weight - compressed_weight)
                rel_error = float((diff_norm / weight_norm).item())
            else:
                rel_error = 0.0

        bias_params = self.module.bias.numel() if self.module.bias is not None else 0
        original_params = self.module.weight.numel() + bias_params
        core_params = compressed.core.numel()
        left_params = compressed.left_basis.numel()
        right_params = compressed.right_basis.numel()
        compressed_params = core_params + left_params + right_params + bias_params

        info.update(
            {
                "original_params": int(original_params),
                "compressed_params": int(compressed_params),
                "core_params": int(core_params),
                "left_basis_params": int(left_params),
                "right_basis_params": int(right_params),
                "compression_ratio": compressed_params / max(1, original_params),
                "frobenius_relative_error": rel_error,
            }
        )
        return compressed, info


def attach_curvature_accumulators(
    model: nn.Module,
    cfg: SubspaceExperimentConfig,
    device: torch.device,
    target_paths: List[str],
) -> Dict[str, CurvatureAccumulator]:
    accumulators: Dict[str, CurvatureAccumulator] = {}
    for path in target_paths:
        module = get_submodule(model, path)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Target module {path} must be nn.Linear, found {type(module)}")
        accumulators[path] = CurvatureAccumulator(
            name=path,
            module=module,
            device=device,
            ema_decay=cfg.curvature_ema_decay,
            min_samples=cfg.curvature_min_samples,
            damping=cfg.curvature_damping,
        )
    return accumulators


def run_training_stage(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
    grad_clip: float,
    device: torch.device,
    stage_name: str,
    apply_updates: bool = True,
) -> None:
    if epochs <= 0:
        return

    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    if apply_updates:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs * len(train_loader))
        )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            if apply_updates and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if apply_updates and optimizer is not None:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                model.zero_grad(set_to_none=True)
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"[{stage_name}] epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}")


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer,
    canaries,
    device: torch.device,
) -> Dict[str, object]:
    train_ppl = evaluate_perplexity(model, train_loader, device)
    val_ppl = evaluate_perplexity(model, val_loader, device)
    canary_metrics = evaluate_canaries(model, tokenizer, canaries, device)
    return {
        "train_perplexity": train_ppl,
        "val_perplexity": val_ppl,
        "canary_metrics": canary_metrics,
    }


def compress_model_with_curvature(
    model: nn.Module,
    accumulators: Dict[str, CurvatureAccumulator],
    cfg: SubspaceExperimentConfig,
) -> List[Dict[str, float]]:
    stats: List[Dict[str, float]] = []
    for name in sorted(accumulators.keys()):
        accumulator = accumulators[name]
        compressed, info = accumulator.build_compressed_layer(
            variance_keep_in=cfg.variance_keep_in,
            variance_keep_out=cfg.variance_keep_out,
            keep_fraction_in=cfg.keep_fraction_in,
            keep_fraction_out=cfg.keep_fraction_out,
        )
        set_submodule(model, name, compressed)
        stats.append(info)
    return stats


def _build_compressed_model(
    reference_model: nn.Module,
    target_paths: List[str],
    accumulators: Dict[str, CurvatureAccumulator],
    keep_fraction_in: Optional[float],
    keep_fraction_out: Optional[float],
    variance_keep_in: float,
    variance_keep_out: float,
    use_pretrained: bool,
    device: torch.device,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Create a deep copy of ``reference_model`` with MLP layers replaced by curvature-subspace
    linear modules computed from ``accumulators`` using the specified keep fractions.
    Returns the eval-ready model (TinyTransformer or GPT2Wrapper) and the collected stats.
    """

    model_copy = copy.deepcopy(reference_model)
    stats: List[Dict[str, float]] = []
    for path in target_paths:
        accumulator = accumulators[path]
        compressed, info = accumulator.build_compressed_layer(
            variance_keep_in=variance_keep_in,
            variance_keep_out=variance_keep_out,
            keep_fraction_in=keep_fraction_in,
            keep_fraction_out=keep_fraction_out,
        )
        set_submodule(model_copy, path, compressed)
        stats.append(info)

    if use_pretrained:
        eval_model = GPT2Wrapper(model_copy)
    else:
        eval_model = model_copy

    eval_model = eval_model.to(device)
    return eval_model, stats


def serialize_config(cfg: SubspaceExperimentConfig) -> Dict[str, object]:
    data = asdict(cfg)
    data["cache_dir"] = str(cfg.cache_dir)
    if cfg.checkpoint_path is not None:
        data["checkpoint_path"] = str(cfg.checkpoint_path)
    if cfg.output is not None:
        data["output"] = str(cfg.output)
    if cfg.output_dir is not None:
        data["output_dir"] = str(cfg.output_dir)
    if cfg.keep_fraction_sweep is not None:
        data["keep_fraction_sweep"] = list(cfg.keep_fraction_sweep)
    return data


def run_experiment(cfg: SubspaceExperimentConfig) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run curvature subspace compression experiments.")
    if not cfg.device.startswith("cuda"):
        raise ValueError("Please specify a CUDA device for this experiment.")

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    train_dataset, val_dataset, canaries, tokenizer = prepare_datasets(cfg, cfg.cache_dir)
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

    use_pretrained = bool(cfg.pretrained_model)
    if use_pretrained:
        print(f"Loading pretrained model: {cfg.pretrained_model}")
        base_model = GPT2LMHeadModel.from_pretrained(cfg.pretrained_model)
        base_model.config.use_cache = False
        if base_model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.resize_token_embeddings(len(tokenizer))
        for block in base_model.transformer.h:  # type: ignore[attr-defined]
            if isinstance(block.mlp.c_fc, Conv1D):
                block.mlp.c_fc = convert_conv1d_to_linear(block.mlp.c_fc)
            if isinstance(block.mlp.c_proj, Conv1D):
                block.mlp.c_proj = convert_conv1d_to_linear(block.mlp.c_proj)
        base_model.to(device)
        model = GPT2Wrapper(base_model)
        target_paths = collect_target_paths(base_model, cfg, use_pretrained=True)
    else:
        base_model = TinyTransformer(cfg).to(device)
        model = base_model
        target_paths = collect_target_paths(base_model, cfg, use_pretrained=False)
        if cfg.checkpoint_path is not None:
            state_dict = torch.load(cfg.checkpoint_path, map_location=device)
            base_model.load_state_dict(state_dict)

    cfg.vocab_size = len(tokenizer)
    if use_pretrained:
        cfg.model_dim = base_model.config.n_embd  # type: ignore[attr-defined]
        cfg.n_layers = base_model.config.n_layer  # type: ignore[attr-defined]
        cfg.n_heads = base_model.config.n_head  # type: ignore[attr-defined]
        inner_dim = getattr(base_model.config, "n_inner", None)  # type: ignore[attr-defined]
        if inner_dim:
            cfg.mlp_mult = max(1, int(round(inner_dim / cfg.model_dim)))

    accumulators = attach_curvature_accumulators(base_model, cfg, device, target_paths)

    run_training_stage(
        model=model,
        train_loader=train_loader,
        epochs=cfg.pretrain_epochs,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        grad_clip=cfg.grad_clip,
        device=device,
        stage_name="pretrain",
        apply_updates=cfg.pretrain_apply_updates,
    )

    baseline_metrics = evaluate_model(model, train_loader, val_loader, tokenizer, canaries, device)

    for accumulator in accumulators.values():
        accumulator.close()

    sweep_values = list(cfg.keep_fraction_sweep) if cfg.keep_fraction_sweep else []
    if sweep_values:
        unique_fractions: List[float] = []
        for value in sweep_values:
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"keep-fractions values must be in (0, 1], got {value}.")
            if not any(math.isclose(value, existing, rel_tol=1e-6, abs_tol=1e-6) for existing in unique_fractions):
                unique_fractions.append(value)

        if not any(math.isclose(1.0, value, rel_tol=1e-6, abs_tol=1e-6) for value in unique_fractions):
            unique_fractions.insert(0, 1.0)
        else:
            unique_fractions = [1.0] + [val for val in unique_fractions if not math.isclose(val, 1.0, rel_tol=1e-6, abs_tol=1e-6)]

        output_dir: Optional[Path] = cfg.output_dir
        if output_dir is None and cfg.output is not None:
            output_dir = cfg.output
            if output_dir.suffix:  # treat file path as directory by using parent
                output_dir = output_dir.parent
        if output_dir is None:
            raise ValueError("An output directory must be specified via --output-dir when using --keep-fractions.")
        if cfg.finetune_epochs > 0:
            raise ValueError("Fine-tuning is not supported when sweeping multiple keep fractions in a single run.")
        output_dir.mkdir(parents=True, exist_ok=True)

        fraction_results: Dict[str, Dict[str, object]] = {}
        for fraction in unique_fractions:
            keep_in = fraction
            keep_out = fraction

            eval_model, compression_stats = _build_compressed_model(
                reference_model=base_model,
                target_paths=target_paths,
                accumulators=accumulators,
                keep_fraction_in=keep_in,
                keep_fraction_out=keep_out,
                variance_keep_in=cfg.variance_keep_in,
                variance_keep_out=cfg.variance_keep_out,
                use_pretrained=use_pretrained,
                device=device,
            )
            compressed_metrics = evaluate_model(eval_model, train_loader, val_loader, tokenizer, canaries, device)

            result_cfg = serialize_config(cfg)
            result_cfg["keep_fraction_in"] = keep_in
            result_cfg["keep_fraction_out"] = keep_out

            fraction_key, _ = _fraction_to_key(fraction)
            result: Dict[str, object] = {
                "config": result_cfg,
                "baseline": copy.deepcopy(baseline_metrics),
                "compression": compression_stats,
                "compressed": compressed_metrics,
                "target_modules": target_paths,
                "used_pretrained_model": cfg.pretrained_model if use_pretrained else None,
                "fraction": fraction,
            }

            output_path = output_dir / f"{fraction_key}.json"
            with output_path.open("w") as handle:
                json.dump(result, handle, indent=2)
            print(f"Saved results to {output_path}")

            fraction_results[fraction_key] = result
            del eval_model

        return {"fraction_results": fraction_results}

    compression_stats = compress_model_with_curvature(base_model, accumulators, cfg)

    compressed_metrics = evaluate_model(model, train_loader, val_loader, tokenizer, canaries, device)

    finetuned_metrics: Optional[Dict[str, object]] = None
    if cfg.finetune_epochs > 0:
        run_training_stage(
            model=model,
            train_loader=train_loader,
            epochs=cfg.finetune_epochs,
            lr=cfg.finetune_lr,
            weight_decay=cfg.finetune_weight_decay,
            betas=cfg.betas,
            grad_clip=cfg.grad_clip,
            device=device,
            stage_name="finetune",
            apply_updates=True,
        )
        finetuned_metrics = evaluate_model(model, train_loader, val_loader, tokenizer, canaries, device)

    results: Dict[str, object] = {
        "config": serialize_config(cfg),
        "baseline": baseline_metrics,
        "compression": compression_stats,
        "compressed": compressed_metrics,
        "target_modules": target_paths,
        "used_pretrained_model": cfg.pretrained_model if use_pretrained else None,
    }
    if finetuned_metrics is not None:
        results["finetuned"] = finetuned_metrics

    if cfg.output:
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {cfg.output}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curvature subspace compression on a tiny transformer.")
    parser.add_argument("--cache-dir", type=Path, default=Path("./data"), help="Directory for cached datasets.")
    parser.add_argument("--pretrained-model", type=str, default=None, help="Hugging Face model id to load instead of training from scratch.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a local TinyTransformer checkpoint (state_dict) to initialize from.")
    parser.add_argument("--pretrain-epochs", type=int, default=1, help="Epochs for the curvature-stat collection stage.")
    parser.add_argument("--finetune-epochs", type=int, default=0, help="Epochs to fine-tune after compression.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the pretraining stage.")
    parser.add_argument("--finetune-lr", type=float, default=1e-4, help="Learning rate for the fine-tuning stage.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay during pretraining.")
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-2, help="Weight decay during fine-tuning.")
    parser.add_argument("--canary-count", type=int, default=16)
    parser.add_argument("--canary-repetitions", type=int, default=50)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-val-tokens", type=int, default=None)
    parser.add_argument("--variance-keep-in", type=float, default=0.9, help="Variance retained along the input curvature eigenvectors.")
    parser.add_argument("--variance-keep-out", type=float, default=0.9, help="Variance retained along the output curvature eigenvectors.")
    parser.add_argument("--curvature-ema-decay", type=float, default=5e-4)
    parser.add_argument("--curvature-min-samples", type=int, default=512)
    parser.add_argument("--curvature-damping", type=float, default=1e-3)
    parser.add_argument("--no-compress-fc1", action="store_true", help="Skip compressing the first MLP projection.")
    parser.add_argument("--no-compress-fc2", action="store_true", help="Skip compressing the second MLP projection.")
    parser.add_argument("--keep-fraction-in", type=float, default=None, help="Fraction of input dimensions to keep (overrides variance target).")
    parser.add_argument("--keep-fraction-out", type=float, default=None, help="Fraction of output dimensions to keep (overrides variance target).")
    parser.add_argument(
        "--keep-fractions",
        type=float,
        nargs="+",
        default=None,
        help="Optional sweep of keep fractions (applied to both input/output dimensions).",
    )
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save results as JSON.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save multiple outputs when using --keep-fractions.")
    parser.add_argument(
        "--no-pretrain-updates",
        action="store_true",
        help="Disable optimizer updates during the pretrain curvature-collection stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SubspaceExperimentConfig(
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        block_size=args.block_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        cache_dir=args.cache_dir,
        pretrained_model=args.pretrained_model,
        checkpoint_path=args.checkpoint,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_weight_decay=args.finetune_weight_decay,
        canary_count=args.canary_count,
        canary_repetitions=args.canary_repetitions,
        max_train_tokens=args.max_train_tokens,
        max_val_tokens=args.max_val_tokens,
        variance_keep_in=args.variance_keep_in,
        variance_keep_out=args.variance_keep_out,
        curvature_ema_decay=args.curvature_ema_decay,
        curvature_min_samples=args.curvature_min_samples,
        curvature_damping=args.curvature_damping,
        compress_fc1=not args.no_compress_fc1,
        compress_fc2=not args.no_compress_fc2,
        keep_fraction_in=args.keep_fraction_in,
        keep_fraction_out=args.keep_fraction_out,
        keep_fraction_sweep=args.keep_fractions,
        device=args.device,
        seed=args.seed,
        output=args.output,
        output_dir=args.output_dir,
        epochs=args.pretrain_epochs,
        pretrain_apply_updates=not args.no_pretrain_updates,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()

