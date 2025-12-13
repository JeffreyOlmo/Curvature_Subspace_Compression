from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from pytorch_optimizer.optimizer.shampoo_utils import compute_power_schur_newton

from .curvature_regularizer import (
    CurvatureRegularizer,
    CurvatureRegularizerConfig,
    ShardedCurvatureRegularizer,
)


@dataclass
class ShampooCurvatureOptions:
    lambda_crt: float = 1e-3
    ema_decay: float = 1e-3
    preconditioning_compute_steps: int = 50
    damping: float = 1e-2
    min_samples: int = 128
    matrix_eps: float = 1e-6
    max_iters: int = 50
    track_stats: bool = False
    shard_size: Optional[int] = None


def _shampoo_inverse(matrix: torch.Tensor, matrix_eps: float, max_iters: int) -> torch.Tensor:
    """Compute matrix inverse using Shampoo's coupled Newton iteration."""
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("Expected a square matrix for Shampoo inverse computation.")
    # Clone to avoid in-place modifications inside the Newton iteration.
    mat = matrix.clone()
    return compute_power_schur_newton(mat, p=1, ridge_epsilon=matrix_eps, max_iters=max_iters)


class ShampooCurvatureRegularizer(CurvatureRegularizer):
    """Curvature regularizer leveraging Shampoo's inverse computation."""

    def __init__(
        self,
        module: nn.Linear,
        config: CurvatureRegularizerConfig,
        matrix_eps: float,
        max_iters: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(module=module, config=config, device=device, inverse_provider=None)
        self._matrix_eps = matrix_eps
        self._max_iters = max_iters

    def _refresh_inverses(self, force: bool = False) -> None:
        if not force and self._samples_seen < self.config.min_samples:
            return

        eye_in = torch.eye(self._dim_in, device=self.device, dtype=self._A.dtype)
        eye_out = torch.eye(self._dim_out, device=self.device, dtype=self._G.dtype)
        damping = self.config.damping

        A_damped = self._A + damping * eye_in
        G_damped = self._G + damping * eye_out

        self._A_inv = _shampoo_inverse(A_damped, self._matrix_eps, self._max_iters)
        self._G_inv = _shampoo_inverse(G_damped, self._matrix_eps, self._max_iters)

        self._inverse_updates += 1

        if self.config.track_stats:
            A_eigs = torch.linalg.eigvalsh(A_damped)
            G_eigs = torch.linalg.eigvalsh(G_damped)
            self._latest_A_eig_min = float(A_eigs.min().item())
            self._latest_A_eig_max = float(A_eigs.max().item())
            self._latest_G_eig_min = float(G_eigs.min().item())
            self._latest_G_eig_max = float(G_eigs.max().item())


class ShampooShardedCurvatureRegularizer(ShardedCurvatureRegularizer):
    """Sharded variant using Shampoo inversion per shard."""

    def __init__(
        self,
        module: nn.Linear,
        config: CurvatureRegularizerConfig,
        shard_size: int,
        matrix_eps: float,
        max_iters: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(module=module, config=config, shard_size=shard_size, device=device, inverse_provider=None)
        self._matrix_eps = matrix_eps
        self._max_iters = max_iters

    def _refresh_inverses(self, force: bool = False) -> None:
        if not force and self._samples_seen < self.config.min_samples:
            return

        eye_in = torch.eye(self._dim_in, device=self.device, dtype=self._A.dtype)
        damping = self.config.damping
        A_damped = self._A + damping * eye_in
        self._A_inv = _shampoo_inverse(A_damped, self._matrix_eps, self._max_iters)

        self._G_inv_shards = []
        for idx, (start, end) in enumerate(self._output_shards):
            shard = self._G_shards[idx]
            eye_out = torch.eye(end - start, device=self.device, dtype=shard.dtype)
            G_damped = shard + damping * eye_out
            G_inv = _shampoo_inverse(G_damped, self._matrix_eps, self._max_iters)
            self._G_inv_shards.append(G_inv)

        self._inverse_updates += 1

        if self.config.track_stats:
            A_eigs = torch.linalg.eigvalsh(A_damped)
            self._latest_A_eig_min = float(A_eigs.min().item())
            self._latest_A_eig_max = float(A_eigs.max().item())
            shard_eig_mins = [
                float(torch.linalg.eigvalsh(self._G_shards[idx]).min().item()) for idx in range(self._shard_count)
            ]
            shard_eig_maxs = [
                float(torch.linalg.eigvalsh(self._G_shards[idx]).max().item()) for idx in range(self._shard_count)
            ]
            self._latest_G_eig_min = min(shard_eig_mins)
            self._latest_G_eig_max = max(shard_eig_maxs)


def attach_shampoo_curvature_regularizer(
    module: nn.Linear,
    lambda_crt: float = 1e-3,
    ema_decay: float = 1e-3,
    preconditioning_compute_steps: int = 50,
    damping: float = 1e-2,
    min_samples: int = 128,
    matrix_eps: float = 1e-6,
    max_iters: int = 50,
    shard_size: Optional[int] = None,
    track_stats: bool = False,
    device: Optional[torch.device] = None,
) -> CurvatureRegularizer:
    """Attach a Shampoo-backed curvature regularizer to a module."""
    base_config = CurvatureRegularizerConfig(
        lambda_crt=lambda_crt,
        ema_decay=ema_decay,
        inv_update_interval=preconditioning_compute_steps,
        damping=damping,
        min_samples=min_samples,
        track_stats=track_stats,
    )

    if shard_size is not None and module.out_features > shard_size:
        return ShampooShardedCurvatureRegularizer(
            module=module,
            config=base_config,
            shard_size=shard_size,
            matrix_eps=matrix_eps,
            max_iters=max_iters,
            device=device,
        )

    return ShampooCurvatureRegularizer(
        module=module,
        config=base_config,
        matrix_eps=matrix_eps,
        max_iters=max_iters,
        device=device,
    )

