import torch
import torch.nn as nn
from typing import Optional


class CurvatureSubspaceLinear(nn.Module):
    """
    Linear layer parameterised by a curvature-aligned subspace.

    The weight matrix is represented as:
        W = L @ C @ R.T
    where:
        - L ∈ ℝ^{out_features × rank_out}   (left basis, fixed)
        - C ∈ ℝ^{rank_out × rank_in}        (learnable core)
        - R ∈ ℝ^{in_features × rank_in}     (right basis, fixed)
    """

    def __init__(
        self,
        left_basis: torch.Tensor,
        core: torch.Tensor,
        right_basis: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        if left_basis.dim() != 2 or right_basis.dim() != 2:
            raise ValueError("left_basis and right_basis must be 2-D tensors.")
        if core.dim() != 2:
            raise ValueError("core must be a 2-D tensor.")

        out_features, rank_out = left_basis.shape
        in_features, rank_in = right_basis.shape

        if core.shape != (rank_out, rank_in):
            raise ValueError(
                f"core shape {tuple(core.shape)} must match (rank_out, rank_in)=({rank_out}, {rank_in})."
            )

        self.out_features = out_features
        self.in_features = in_features
        self.rank_out = rank_out
        self.rank_in = rank_in

        self.register_buffer("left_basis", left_basis)
        self.register_buffer("right_basis", right_basis)

        self.core = nn.Parameter(core)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().detach())
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        left_basis: torch.Tensor,
        right_basis: torch.Tensor,
    ) -> "CurvatureSubspaceLinear":
        """
        Construct a curvature subspace layer from a dense nn.Linear module.
        """
        if not isinstance(linear, nn.Linear):
            raise TypeError("Expected an nn.Linear module.")

        device = linear.weight.device
        dtype = linear.weight.dtype

        left = left_basis.to(device=device, dtype=dtype)
        right = right_basis.to(device=device, dtype=dtype)

        with torch.no_grad():
            core = (left.t() @ linear.weight.to(dtype=dtype)) @ right
            bias = linear.bias.clone().detach() if linear.bias is not None else None

        return cls(left_basis=left, core=core, right_basis=right, bias=bias)

    def materialize_weight(self) -> torch.Tensor:
        """
        Return the dense weight matrix corresponding to this layer.
        """
        return self.left_basis @ self.core @ self.right_basis.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.size(-1))

        projected = x_2d @ self.right_basis
        hidden = projected @ self.core.t()
        output = hidden @ self.left_basis.t()

        if self.bias is not None:
            output = output + self.bias

        return output.reshape(*orig_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank_out={self.rank_out}, rank_in={self.rank_in}, "
            f"bias={self.bias is not None}"
        )

