import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from curvature_subspace_compression.memorization_kfac.experiments.run_crt_tiny_transformer import (
    ExperimentConfig,
    TinyTransformer,
    collate_batch,
    prepare_datasets,
)
from curvature_subspace_compression.memorization_kfac.experiments.run_curvature_subspace_tiny_transformer import (
    SubspaceExperimentConfig,
    attach_curvature_accumulators,
    collect_target_paths,
    run_training_stage,
    set_seed,
    set_submodule,
)
from curvature_subspace_compression.memorization_kfac.subspace_linear import CurvatureSubspaceLinear


def _coerce_cfg(dataclass_type, raw: Dict[str, Any]):
    allowed = {f.name for f in fields(dataclass_type)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return dataclass_type(**filtered)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this API server.")
    return torch.device("cuda")


def _load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    # Prefer weights-only loading when possible (PyTorch >= 2.0 supports this kwarg).
    try:
        data = torch.load(path, map_location=device, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        data = torch.load(path, map_location=device)
    if not isinstance(data, dict):
        raise TypeError(f"Checkpoint {path} did not contain a state_dict dict.")
    return data


def _build_tiny_model(cfg: ExperimentConfig, state_dict: Dict[str, torch.Tensor], device: torch.device) -> TinyTransformer:
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_tiny_model_for_state_dict(
    cfg: ExperimentConfig,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> TinyTransformer:
    """
    Build a TinyTransformer whose module structure matches the provided state_dict.

    If the state_dict contains curvature-subspace keys (e.g. `blocks.0.mlp.fc1.core`),
    we replace those layers with `CurvatureSubspaceLinear` modules before loading.
    """
    model = TinyTransformer(cfg).to(device)
    target_paths = collect_target_paths(model, cfg, use_pretrained=False)

    for path in target_paths:
        core_key = f"{path}.core"
        left_key = f"{path}.left_basis"
        right_key = f"{path}.right_basis"
        bias_key = f"{path}.bias"
        if core_key not in state_dict:
            continue
        if left_key not in state_dict or right_key not in state_dict:
            raise KeyError(f"Compressed checkpoint missing {left_key} or {right_key} for layer {path}.")
        core = state_dict[core_key].to(device=device)
        left = state_dict[left_key].to(device=device)
        right = state_dict[right_key].to(device=device)
        bias = state_dict.get(bias_key, None)
        if bias is not None:
            bias = bias.to(device=device)
        compressed_layer = CurvatureSubspaceLinear(left_basis=left, core=core, right_basis=right, bias=bias)
        set_submodule(model, path, compressed_layer)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    device = next(model.parameters()).device
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        ctx = input_ids[:, -model.config.block_size :]  # type: ignore[attr-defined]
        logits, _ = model(ctx, targets=None)  # type: ignore[misc]
        next_logits = logits[:, -1, :]
        if temperature <= 0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / float(temperature)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_logits, min(int(top_k), next_logits.size(-1)))
                cutoff = v[:, -1].unsqueeze(-1)
                next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, -1e10), next_logits)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return tokenizer.decode(input_ids[0].tolist())


class TinyTransformerPair:
    """
    Holds a baseline TinyTransformer checkpoint and a corresponding keep-20% compressed model.

    If a compressed checkpoint is not available, it will be created lazily by re-collecting
    curvature stats (no optimizer updates) and applying the curvature-subspace MLP replacement.
    """

    def __init__(
        self,
        *,
        comp_json_path: Path,
        baseline_ckpt_path: Path,
        cache_dir: Path,
        compressed_ckpt_path: Optional[Path] = None,
        seed: int = 2025,
    ) -> None:
        self.comp_json_path = comp_json_path
        self.baseline_ckpt_path = baseline_ckpt_path
        self.cache_dir = cache_dir
        self.compressed_ckpt_path = compressed_ckpt_path
        self.seed = seed

        self.device = _require_cuda()
        self._tokenizer = None

        self._baseline: Optional[TinyTransformer] = None
        self._compressed: Optional[TinyTransformer] = None
        self._cfg: Optional[SubspaceExperimentConfig] = None

    def load(self) -> None:
        if self._baseline is not None:
            return

        with self.comp_json_path.open() as f:
            comp_data = json.load(f)
        raw_cfg = comp_data.get("config", {})

        # Ensure we use cuda in the serving environment.
        raw_cfg["device"] = "cuda"

        cfg = _coerce_cfg(SubspaceExperimentConfig, raw_cfg)
        cfg.seed = self.seed
        cfg.pretrain_apply_updates = False
        self._cfg = cfg

        set_seed(cfg.seed)
        state_dict = _load_state_dict(self.baseline_ckpt_path, self.device)

        # Tokenizer is GPT-2.
        _, _, _, tokenizer = prepare_datasets(cfg, self.cache_dir)
        self._tokenizer = tokenizer

        self._baseline = _build_tiny_model(cfg, state_dict, self.device)

        # If we have a pre-exported compressed checkpoint, load it.
        if self.compressed_ckpt_path and self.compressed_ckpt_path.exists():
            comp_state = _load_state_dict(self.compressed_ckpt_path, self.device)
            self._compressed = _build_tiny_model_for_state_dict(cfg, comp_state, self.device)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def baseline(self) -> TinyTransformer:
        if self._baseline is None:
            self.load()
        assert self._baseline is not None
        return self._baseline

    def compressed(self) -> TinyTransformer:
        if self._compressed is not None:
            return self._compressed
        self._compressed = self._build_compressed_from_baseline()
        return self._compressed

    def _build_compressed_from_baseline(self) -> TinyTransformer:
        if self._cfg is None:
            self.load()
        assert self._cfg is not None

        # If a checkpoint path is provided but missing, we'll still build and then write it.
        cfg = self._cfg
        base_model = self.baseline

        # Build a fresh copy for curvature collection to avoid contaminating inference modules.
        ref = TinyTransformer(cfg).to(self.device)
        ref.load_state_dict(base_model.state_dict())
        ref.train()

        train_dataset, _, _, _ = prepare_datasets(cfg, self.cache_dir)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_batch,
        )

        target_paths = collect_target_paths(ref, cfg, use_pretrained=False)
        accumulators = attach_curvature_accumulators(ref, cfg, self.device, target_paths)

        t0 = time.time()
        run_training_stage(
            model=ref,
            train_loader=train_loader,
            epochs=int(getattr(cfg, "pretrain_epochs", 1)),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            grad_clip=cfg.grad_clip,
            device=self.device,
            stage_name="curvature_collect",
            apply_updates=False,
        )
        for acc in accumulators.values():
            acc.close()

        # Apply compression to a deep copy (we avoid importing the private helper).
        model_copy = TinyTransformer(cfg).to(self.device)
        model_copy.load_state_dict(base_model.state_dict())
        model_copy.eval()

        for path in target_paths:
            compressed_layer, _info = accumulators[path].build_compressed_layer(
                variance_keep_in=cfg.variance_keep_in,
                variance_keep_out=cfg.variance_keep_out,
                keep_fraction_in=cfg.keep_fraction_in,
                keep_fraction_out=cfg.keep_fraction_out,
            )
            set_submodule(model_copy, path, compressed_layer)

        dt = time.time() - t0
        print(f"[api] built compressed model in {dt:.1f}s (keep={cfg.keep_fraction_in})")

        if self.compressed_ckpt_path:
            self.compressed_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_copy.state_dict(), self.compressed_ckpt_path)
            print(f"[api] saved compressed checkpoint to {self.compressed_ckpt_path}")

        return model_copy


