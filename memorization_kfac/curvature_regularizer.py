import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from pytorch_optimizer.optimizer.shampoo_utils import compute_power_svd


@dataclass
class CurvatureRegularizerConfig:
    """Configuration for curvature-aware weight decay."""

    lambda_crt: float = 1e-3
    ema_decay: float = 1e-3
    inv_update_interval: int = 50
    damping: float = 1e-2
    min_samples: int = 128
    track_stats: bool = False


class CurvatureRegularizer:
    """Per-step curvature-aware weight decay for a single linear module.

    Maintains exponential-moving-average estimates of activation and gradient
    covariances (A and G) and injects the gradient `lambda * G^{-1} W A^{-1}`
    into the module's weight gradient each training step. The curvature stats
    are refreshed every `inv_update_interval` steps to keep the update stable
    without having to materialize eigenbases or run heavy decompositions.
    """

    def __init__(
        self,
        module: nn.Linear,
        config: CurvatureRegularizerConfig,
        device: Optional[torch.device] = None,
        inverse_provider: Optional[
            Callable[[nn.Linear, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> None:
        if not isinstance(module, nn.Linear):
            raise TypeError("CurvatureRegularizer currently supports nn.Linear modules only.")

        self.module = module
        self.config = config
        self.device = device or module.weight.device
        self._inverse_provider = inverse_provider

        self._dim_out, self._dim_in = module.weight.shape
        self._A = torch.eye(self._dim_in, device=self.device)
        self._G = torch.eye(self._dim_out, device=self.device)
        self._A_inv = torch.eye(self._dim_in, device=self.device)
        self._G_inv = torch.eye(self._dim_out, device=self.device)
        self._tracking_started = False
        self._step = 0
        self._samples_seen = 0
        self._last_activation: Optional[torch.Tensor] = None

        self._metric_reg_norm_sum = 0.0
        self._metric_grad_norm_sum = 0.0
        self._metric_ratio_sum = 0.0
        self._metric_calls = 0
        self._last_reg_norm = math.nan
        self._last_ratio = math.nan
        self._metric_ratio_max = 0.0
        self._metric_ratio_min = math.inf
        self._latest_A_eig_min = math.nan
        self._latest_A_eig_max = math.nan
        self._latest_G_eig_min = math.nan
        self._latest_G_eig_max = math.nan
        self._inverse_updates = 0
        self._shard_count = 1
        self._output_shards: Optional[List[Tuple[int, int]]] = None
        self._G_inv_shards: Optional[List[torch.Tensor]] = None

        self._forward_handle = module.register_forward_pre_hook(self._forward_hook, with_kwargs=False)
        self._backward_handle = module.register_full_backward_hook(self._backward_hook, prepend=False)

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    # --------------------------------------------------------------------- #
    # Hooks                                                                 #
    # --------------------------------------------------------------------- #

    def _forward_hook(self, module: nn.Module, inputs) -> None:
        if not module.training or self.config.lambda_crt <= 0 or self._A_inv is None or self._G_inv is None:
            self._last_activation = None
            return

        if not inputs:
            self._last_activation = None
            return

        activation = inputs[0]
        if activation is None:
            self._last_activation = None
            return

        if activation.dim() > 2:
            activation = activation.reshape(-1, activation.size(-1))
        self._last_activation = activation.detach().to(self.device, dtype=torch.float32)

    def _backward_hook(self, module: nn.Module, grad_input, grad_output) -> None:
        if not module.training or self.config.lambda_crt <= 0:
            self._last_activation = None
            return

        if self._last_activation is None or not grad_output:
            self._last_activation = None
            return

        grad_out = grad_output[0]
        if grad_out is None:
            self._last_activation = None
            return

        grad_out = grad_out.detach().to(self.device, dtype=torch.float32)
        if grad_out.dim() > 2:
            grad_out = grad_out.reshape(-1, grad_out.size(-1))

        batch_size = grad_out.size(0)
        if batch_size == 0:
            self._last_activation = None
            return

        act = self._last_activation
        self._update_covariances(act, grad_out)
        self._apply_regularizer()

        self._last_activation = None

    # --------------------------------------------------------------------- #
    # Covariance updates                                                    #
    # --------------------------------------------------------------------- #

    def _update_covariances(self, act: torch.Tensor, grad_out: torch.Tensor) -> None:
        """Update EMA estimates for A and G."""
        rho = self.config.ema_decay
        batch_size = act.size(0)
        norm_factor = 1.0 / float(batch_size)

        A_batch = (act.t() @ act) * norm_factor
        G_batch = (grad_out.t() @ grad_out) * norm_factor

        if not self._tracking_started:
            self._A.copy_(A_batch + self.config.damping * torch.eye(self._dim_in, device=self.device))
            self._G.copy_(G_batch + self.config.damping * torch.eye(self._dim_out, device=self.device))
            self._tracking_started = True
            self._refresh_inverses(force=True)
        else:
            self._A.mul_(1.0 - rho).add_(A_batch, alpha=rho)
            self._G.mul_(1.0 - rho).add_(G_batch, alpha=rho)

        self._samples_seen += batch_size
        self._step += 1

        if (
            self._samples_seen >= self.config.min_samples
            and self._step % self.config.inv_update_interval == 0
        ):
            self._refresh_inverses()

    def _refresh_inverses(self, force: bool = False) -> None:
        """Recompute and cache damped inverses for A and G."""
        if not force and self._samples_seen < self.config.min_samples:
            return

        if self._inverse_provider is not None:
            A_inv, G_inv = self._inverse_provider(self.module, self._A, self._G)
            self._A_inv = A_inv.to(self.device)
            self._G_inv = G_inv.to(self.device)
        else:
            eye_in = torch.eye(self._dim_in, device=self.device)
            eye_out = torch.eye(self._dim_out, device=self.device)
            damping = self.config.damping

            A_damped = self._A + damping * eye_in
            G_damped = self._G + damping * eye_out

            self._A_inv = compute_power_svd(A_damped, 1)
            self._G_inv = compute_power_svd(G_damped, 1)
        self._inverse_updates += 1

        if self.config.track_stats:
            A_eigs = torch.linalg.eigvalsh(A_damped)
            G_eigs = torch.linalg.eigvalsh(G_damped)
            self._latest_A_eig_min = float(A_eigs.min().item())
            self._latest_A_eig_max = float(A_eigs.max().item())
            self._latest_G_eig_min = float(G_eigs.min().item())
            self._latest_G_eig_max = float(G_eigs.max().item())

    # --------------------------------------------------------------------- #
    # Regularizer                                                           #
    # --------------------------------------------------------------------- #

    def _apply_regularizer(self) -> None:
        """Inject curvature-aware weight decay into the module's gradient."""
        weight = self.module.weight
        if weight.grad is None:
            return
        if self._A_inv is None or self._G_inv is None:
            return

        with torch.no_grad():
            if self._samples_seen < self.config.min_samples:
                return
            reg_grad = self._compute_regularizer_gradient(weight.detach())
            if reg_grad is None:
                return
            if self.config.track_stats:
                reg_norm = torch.linalg.vector_norm(reg_grad).item()
                grad_norm = torch.linalg.vector_norm(weight.grad).item()
                ratio = reg_norm / (grad_norm + 1e-12)
                self._metric_reg_norm_sum += reg_norm
                self._metric_grad_norm_sum += grad_norm
                self._metric_ratio_sum += ratio
                self._metric_calls += 1
                self._last_reg_norm = reg_norm
                self._last_ratio = ratio
                self._metric_ratio_max = max(self._metric_ratio_max, ratio)
                self._metric_ratio_min = min(self._metric_ratio_min, ratio)
            weight.grad.add_(reg_grad, alpha=self.config.lambda_crt)

    def _compute_regularizer_gradient(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        if self._G_inv is None or self._A_inv is None:
            return None
        return self._G_inv @ weight @ self._A_inv

    def low_curvature_energy(self, fraction: float = 0.5) -> Optional[dict]:
        """Return squared weight energy within the lowest-curvature subspace.

        The curvature basis is defined by the eigenvectors of the damped
        activation and gradient covariances. Energy is measured as the squared
        Frobenius norm of the weight projected onto the outer-product basis of
        the bottom `fraction` of eigenvalue pairs (smallest λ_G * λ_A).
        """
        if fraction <= 0.0 or fraction >= 1.0:
            raise ValueError("fraction must be in (0, 1).")
        if not self._tracking_started or self._A is None or self._G is None:
            return None
        if self._samples_seen < self.config.min_samples:
            return None

        dim_in = self._dim_in
        dim_out = self._dim_out
        damping = self.config.damping

        with torch.no_grad():
            eye_in = torch.eye(dim_in, device=self._A.device, dtype=self._A.dtype)
            eye_out = torch.eye(dim_out, device=self._G.device, dtype=self._G.dtype)

            A_damped = (self._A + damping * eye_in).detach().to(dtype=torch.float64, device="cpu")
            G_damped = (self._G + damping * eye_out).detach().to(dtype=torch.float64, device="cpu")

            try:
                eigvals_A, eigvecs_A = torch.linalg.eigh(A_damped)
                eigvals_G, eigvecs_G = torch.linalg.eigh(G_damped)
            except RuntimeError as err:
                raise RuntimeError(f"Failed to compute curvature eigenbasis: {err}") from err

            weight = self.module.weight.detach().to(dtype=torch.float64, device="cpu")
            coeff = eigvecs_G.t().mm(weight).mm(eigvecs_A)

            prod_eigs = torch.outer(eigvals_G, eigvals_A).reshape(-1)
            coeff_flat = coeff.reshape(-1)

            total_energy = float(torch.sum(coeff_flat * coeff_flat).item())
            if total_energy <= 0.0:
                return {
                    "fraction": fraction,
                    "low_energy": 0.0,
                    "total_energy": 0.0,
                    "ratio": 0.0,
                    "eigvals_A_min": float(eigvals_A.min().item()),
                    "eigvals_A_max": float(eigvals_A.max().item()),
                    "eigvals_G_min": float(eigvals_G.min().item()),
                    "eigvals_G_max": float(eigvals_G.max().item()),
                }

            n_pairs = prod_eigs.numel()
            k = max(1, int(fraction * n_pairs))
            _, indices = torch.topk(prod_eigs, k, largest=False)
            low_coeff = coeff_flat.index_select(0, indices)
            low_energy = float(torch.sum(low_coeff * low_coeff).item())
            ratio = low_energy / total_energy if total_energy > 0.0 else 0.0

            return {
                "fraction": fraction,
                "low_energy": low_energy,
                "total_energy": total_energy,
                "ratio": ratio,
                "eigvals_A_min": float(eigvals_A.min().item()),
                "eigvals_A_max": float(eigvals_A.max().item()),
                "eigvals_G_min": float(eigvals_G.min().item()),
                "eigvals_G_max": float(eigvals_G.max().item()),
            }

    # --------------------------------------------------------------------- #
    # Convenience                                                           #
    # --------------------------------------------------------------------- #

    def state_dict(self) -> dict:
        return {
            "A": self._A,
            "G": self._G,
            "A_inv": self._A_inv,
            "G_inv": self._G_inv,
            "step": self._step,
            "samples_seen": self._samples_seen,
            "inverse_updates": self._inverse_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self._A.copy_(state["A"])
        self._G.copy_(state["G"])
        self._A_inv.copy_(state["A_inv"])
        self._G_inv.copy_(state["G_inv"])
        self._step = state["step"]
        self._samples_seen = state["samples_seen"]
        self._inverse_updates = state.get("inverse_updates", 0)
        self._tracking_started = True

    def collect_metrics(self) -> dict:
        """Return and reset running statistic summaries."""
        calls = self._metric_calls
        avg_ratio = self._metric_ratio_sum / calls if calls > 0 else 0.0
        avg_reg = self._metric_reg_norm_sum / calls if calls > 0 else 0.0
        avg_grad = self._metric_grad_norm_sum / calls if calls > 0 else 0.0
        max_ratio = self._metric_ratio_max if calls > 0 else 0.0
        min_ratio = self._metric_ratio_min if calls > 0 else 0.0
        if calls == 0:
            min_ratio = 0.0

        metrics = {
            "applied_steps": calls,
            "avg_reg_grad_norm": avg_reg,
            "avg_grad_norm": avg_grad,
            "avg_ratio": avg_ratio,
            "max_ratio": max_ratio,
            "min_ratio": min_ratio,
            "last_reg_grad_norm": self._last_reg_norm,
            "last_ratio": self._last_ratio,
            "samples_seen": self._samples_seen,
            "inverse_updates": self._inverse_updates,
            "latest_A_eig_min": self._latest_A_eig_min,
            "latest_A_eig_max": self._latest_A_eig_max,
            "latest_G_eig_min": self._latest_G_eig_min,
            "latest_G_eig_max": self._latest_G_eig_max,
        }

        self._metric_reg_norm_sum = 0.0
        self._metric_grad_norm_sum = 0.0
        self._metric_ratio_sum = 0.0
        self._metric_calls = 0
        self._metric_ratio_max = 0.0
        self._metric_ratio_min = math.inf
        return metrics

def attach_curvature_regularizer(
    module: nn.Linear,
    lambda_crt: float = 1e-3,
    ema_decay: float = 1e-3,
    inv_update_interval: int = 50,
    damping: float = 1e-2,
    min_samples: int = 128,
    device: Optional[torch.device] = None,
    track_stats: bool = True,
    inverse_provider: Optional[
        Callable[[nn.Linear, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
    shard_size: Optional[int] = None,
) -> CurvatureRegularizer:
    """Helper to construct and attach a curvature regularizer to a module."""
    config = CurvatureRegularizerConfig(
        lambda_crt=lambda_crt,
        ema_decay=ema_decay,
        inv_update_interval=inv_update_interval,
        damping=damping,
        min_samples=min_samples,
        track_stats=track_stats,
    )
    if shard_size is not None and module.out_features > shard_size:
        return ShardedCurvatureRegularizer(
            module=module,
            config=config,
            shard_size=shard_size,
            device=device,
            inverse_provider=inverse_provider,
        )
    return CurvatureRegularizer(
        module=module,
        config=config,
        device=device,
        inverse_provider=inverse_provider,
    )


class ShardedCurvatureRegularizer(CurvatureRegularizer):
    """Shard-aware variant to handle wide output matrices (e.g., LM head)."""

    def __init__(
        self,
        module: nn.Linear,
        config: CurvatureRegularizerConfig,
        shard_size: int = 4096,
        device: Optional[torch.device] = None,
        inverse_provider: Optional[
            Callable[[nn.Linear, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> None:
        super().__init__(module=module, config=config, device=device, inverse_provider=inverse_provider)
        if shard_size <= 0:
            raise ValueError("shard_size must be positive.")
        self._shard_size = shard_size
        self._output_shards = self._build_output_shards(self._dim_out, shard_size)
        self._shard_count = len(self._output_shards)
        self._G_inv_shards = [torch.eye(end - start, device=self.device) for start, end in self._output_shards]
        self._G_shards = [torch.eye(end - start, device=self.device) for start, end in self._output_shards]
        self._G = None  # do not materialize the full matrix

    def low_curvature_energy(self, fraction: float = 0.5) -> Optional[dict]:
        raise NotImplementedError("Low-curvature energy tracing is not implemented for sharded CRT modules.")

    @staticmethod
    def _build_output_shards(dim_out: int, shard_size: int) -> List[Tuple[int, int]]:
        shards: List[Tuple[int, int]] = []
        start = 0
        while start < dim_out:
            end = min(start + shard_size, dim_out)
            shards.append((start, end))
            start = end
        return shards

    def _update_covariances(self, act: torch.Tensor, grad_out: torch.Tensor) -> None:
        """EMA updates treating G as block-diagonal shards only."""
        rho = self.config.ema_decay
        batch_size = act.size(0)
        norm_factor = 1.0 / float(batch_size)

        A_batch = (act.t() @ act) * norm_factor

        if not self._tracking_started:
            self._A.copy_(A_batch + self.config.damping * torch.eye(self._dim_in, device=self.device))
            for idx, (start, end) in enumerate(self._output_shards):
                grad_shard = grad_out[:, start:end]
                G_batch = (grad_shard.t() @ grad_shard) * norm_factor
                self._G_shards[idx] = G_batch + self.config.damping * torch.eye(end - start, device=self.device)
            self._tracking_started = True
            self._refresh_inverses(force=True)
        else:
            self._A.mul_(1.0 - rho).add_(A_batch, alpha=rho)
            for idx, (start, end) in enumerate(self._output_shards):
                grad_shard = grad_out[:, start:end]
                G_batch = (grad_shard.t() @ grad_shard) * norm_factor
                self._G_shards[idx].mul_(1.0 - rho).add_(G_batch, alpha=rho)

        self._samples_seen += batch_size
        self._step += 1

        if (
            self._samples_seen >= self.config.min_samples
            and self._step % self.config.inv_update_interval == 0
        ):
            self._refresh_inverses()

    def _refresh_inverses(self, force: bool = False) -> None:
        if not force and self._samples_seen < self.config.min_samples:
            return
        if self._inverse_provider is not None:
            super()._refresh_inverses(force=force)
            return

        eye_in = torch.eye(self._dim_in, device=self.device)
        damping = self.config.damping
        A_damped = self._A + damping * eye_in
        self._A_inv = torch.linalg.solve(A_damped, eye_in)

        self._G_inv_shards = []
        for idx, (start, end) in enumerate(self._output_shards):
            shard = self._G_shards[idx]
            G_damped = shard + damping * torch.eye(end - start, device=self.device)
            G_inv = torch.linalg.solve(G_damped, torch.eye(end - start, device=self.device))
            self._G_inv_shards.append(G_inv)

        self._inverse_updates += 1
        if self.config.track_stats:
            A_eigs = torch.linalg.eigvalsh(A_damped)
            self._latest_A_eig_min = float(A_eigs.min().item())
            self._latest_A_eig_max = float(A_eigs.max().item())
            shard_eig_mins = [torch.linalg.eigvalsh(self._G_shards[idx]).min().item() for idx in range(self._shard_count)]
            shard_eig_maxs = [torch.linalg.eigvalsh(self._G_shards[idx]).max().item() for idx in range(self._shard_count)]
            self._latest_G_eig_min = float(min(shard_eig_mins))
            self._latest_G_eig_max = float(max(shard_eig_maxs))

    def _compute_regularizer_gradient(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        if self._G_inv_shards is None or self._output_shards is None:
            return None
        reg_grad = torch.zeros_like(weight)
        for (start, end), G_inv in zip(self._output_shards, self._G_inv_shards):
            weight_shard = weight[start:end]
            reg_grad[start:end] = G_inv @ weight_shard @ self._A_inv
        return reg_grad

    def state_dict(self) -> dict:
        return {
            "A": self._A,
            "G_shards": self._G_shards,
            "A_inv": self._A_inv,
            "G_inv_shards": self._G_inv_shards,
            "step": self._step,
            "samples_seen": self._samples_seen,
            "inverse_updates": self._inverse_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self._A.copy_(state["A"])
        self._G_shards = [shard.to(self.device) for shard in state["G_shards"]]
        self._A_inv.copy_(state["A_inv"])
        self._G_inv_shards = [inv.to(self.device) for inv in state["G_inv_shards"]]
        self._step = state["step"]
        self._samples_seen = state["samples_seen"]
        self._inverse_updates = state.get("inverse_updates", 0)
        self._tracking_started = True

