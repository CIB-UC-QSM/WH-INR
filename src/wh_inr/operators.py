"""Fourier dipole and finite-difference operators used by WH-INR."""

from collections.abc import Sequence
import math

import torch
from torch import Tensor

SPATIAL_DIMS = (-3, -2, -1)


def _triple(name: str, values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values in z, y, x order")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def build_dipole_kernel(
    shape: Sequence[int],
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    b0_direction: Sequence[float] = (0.0, 0.0, 1.0),
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build the unthresholded continuous QSM dipole kernel.

    Spatial axes and metadata use ``(z, y, x)`` order. FFT DC is at index zero,
    matching :func:`torch.fft.fftn`. The returned real kernel has ``D[0]=0``.
    """

    shape = tuple(int(size) for size in shape)
    if len(shape) != 3 or any(size < 2 for size in shape):
        raise ValueError("shape must contain three sizes greater than one")
    spacing = _triple("voxel_size", voxel_size)
    if any(value <= 0.0 for value in spacing):
        raise ValueError("voxel_size must be positive")
    direction = torch.tensor(
        _triple("b0_direction", b0_direction), device=device, dtype=dtype
    )
    norm = torch.linalg.vector_norm(direction)
    if float(norm) == 0.0:
        raise ValueError("b0_direction must be nonzero")
    direction = direction / norm

    frequencies = [
        torch.fft.fftfreq(size, d=delta, device=device, dtype=dtype)
        for size, delta in zip(shape, spacing, strict=True)
    ]
    kz, ky, kx = torch.meshgrid(*frequencies, indexing="ij")
    squared_norm = kz.square() + ky.square() + kx.square()
    projection = kz * direction[0] + ky * direction[1] + kx * direction[2]
    kernel = torch.full(shape, 1.0 / 3.0, device=device, dtype=dtype)
    nonzero = squared_norm > 0
    kernel[nonzero] -= projection[nonzero].square() / squared_norm[nonzero]
    kernel[0, 0, 0] = 0.0
    return kernel


def dipole_forward(chi: Tensor, kernel: Tensor) -> Tensor:
    """Apply ``F^H D F`` to one volume or a batch of volumes."""

    if chi.ndim < 3:
        raise ValueError("chi must have at least three spatial dimensions")
    if tuple(kernel.shape[-3:]) != tuple(chi.shape[-3:]):
        raise ValueError("kernel and chi must have matching spatial shapes")
    spectrum = torch.fft.fftn(chi, dim=SPATIAL_DIMS)
    return torch.fft.ifftn(spectrum * kernel, dim=SPATIAL_DIMS).real


def positive_laplacian(volume: Tensor) -> Tensor:
    """Apply the positive periodic six-neighbour Laplacian ``-Delta``.

    Its sign does not change the squared weak-harmonic penalty. This convention
    makes the operator positive semidefinite and self-adjoint.
    """

    if volume.ndim < 3:
        raise ValueError("volume must have at least three spatial dimensions")
    return sum(
        2.0 * volume
        - torch.roll(volume, shifts=1, dims=axis)
        - torch.roll(volume, shifts=-1, dims=axis)
        for axis in SPATIAL_DIMS
    )

