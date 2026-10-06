"""SIREN network and physical coordinate construction."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class SineLayer(nn.Module):
    """Linear layer followed by the sinusoidal SIREN activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        omega: float,
        first: bool = False,
    ) -> None:
        super().__init__()
        self.omega = float(omega)
        self.linear = nn.Linear(in_features, out_features)
        bound = 1.0 / in_features if first else math.sqrt(6.0 / in_features) / omega
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, coordinates: Tensor) -> Tensor:
        return torch.sin(self.omega * self.linear(coordinates))


class Siren(nn.Module):
    """Coordinate MLP representing the susceptibility volume."""

    def __init__(
        self,
        *,
        in_features: int = 3,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
        hidden_omega: float = 30.0,
    ) -> None:
        super().__init__()
        if hidden_features < 1 or hidden_layers < 0:
            raise ValueError("hidden_features must be positive and hidden_layers nonnegative")
        if omega_0 <= 0.0 or hidden_omega <= 0.0:
            raise ValueError("SIREN frequencies must be positive")
        layers: list[nn.Module] = [
            SineLayer(
                in_features,
                hidden_features,
                omega=omega_0,
                first=True,
            )
        ]
        layers.extend(
            SineLayer(
                hidden_features,
                hidden_features,
                omega=hidden_omega,
            )
            for _ in range(hidden_layers)
        )
        final = nn.Linear(hidden_features, 1)
        bound = math.sqrt(6.0 / hidden_features) / hidden_omega
        nn.init.uniform_(final.weight, -bound, bound)
        nn.init.uniform_(final.bias, -bound, bound)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates: Tensor) -> Tensor:
        return self.network(coordinates)


def coordinate_grid(
    shape: Sequence[int],
    voxel_size: Sequence[float],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Return flattened physical coordinates scaled to the range ``[-1, 1]``."""

    shape = tuple(int(size) for size in shape)
    spacing = tuple(float(value) for value in voxel_size)
    if len(shape) != 3 or len(spacing) != 3:
        raise ValueError("shape and voxel_size must each contain three values")
    extents = [(size - 1) * delta / 2.0 for size, delta in zip(shape, spacing, strict=True)]
    scale = max(extents)
    if scale <= 0.0:
        raise ValueError("coordinate extent must be positive")
    axes = [
        (torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0)
        * delta
        / scale
        for size, delta in zip(shape, spacing, strict=True)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)

