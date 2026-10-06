"""COSMOS total-field simulation with mask-derived exterior sources."""

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage
import torch
from torch import Tensor

from wh_inr.operators import dipole_forward


@dataclass(frozen=True)
class CosmosSimulation:
    field: Tensor
    local_field: Tensor
    outer_field: Tensor
    noise: Tensor
    dilated_mask: Tensor
    exterior_source: Tensor
    noise_std: float
    snr: float
    dilation_radius: int
    outer_scale: float


def spherical_structuring_element(radius: int) -> np.ndarray:
    """Return a discrete 3D Euclidean sphere with the requested radius."""

    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    coordinates = np.ogrid[
        tuple(slice(-radius, radius + 1) for _ in range(3))
    ]
    squared_radius = sum(axis.astype(np.float32) ** 2 for axis in coordinates)
    return np.asarray(squared_radius <= radius**2, dtype=bool)


def dilate_mask(mask: Tensor, radius: int = 5) -> Tensor:
    """Dilate a 3D mask with a spherical structuring element on its source device."""

    mask = torch.as_tensor(mask)
    if mask.ndim != 3:
        raise ValueError("mask must be a three-dimensional volume")
    source = mask.detach().cpu().numpy() > 0
    dilated = ndimage.binary_dilation(
        source,
        structure=spherical_structuring_element(radius),
        border_value=0,
    )
    return torch.from_numpy(dilated).to(device=mask.device, dtype=torch.float32)


@torch.no_grad()
def simulate_cosmos(
    chi: Tensor,
    mask: Tensor,
    dipole_kernel: Tensor,
    *,
    snr: float = 100.0,
    seed: int = 0,
    dilation_radius: int = 5,
    outer_scale: float = 10.0,
) -> CosmosSimulation:
    """Simulate the requested noisy COSMOS total field.

    ``Md`` is the spherical dilation of ``mask``. The exterior source is
    ``1 - Md``, and the outer field is exactly
    ``outer_scale * mask * F^H D F (1 - Md)``. Gaussian noise has standard
    deviation ``max(abs(chi)) / snr`` and is retained only inside the mask.
    """

    chi = torch.as_tensor(chi)
    mask = torch.as_tensor(mask, device=chi.device)
    dipole_kernel = torch.as_tensor(dipole_kernel, device=chi.device)
    if chi.ndim != 3 or mask.shape != chi.shape or dipole_kernel.shape != chi.shape:
        raise ValueError("chi, mask, and dipole_kernel must be matching 3D volumes")
    if chi.is_complex() or mask.is_complex() or not bool(torch.isfinite(chi).all()):
        raise ValueError("chi and mask must be real and finite")
    if not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()):
        raise ValueError("mask must be finite and nonnegative")
    if not bool((mask > 0).any()):
        raise ValueError("mask must be nonempty")
    if not math.isfinite(snr) or snr <= 0.0:
        raise ValueError("snr must be finite and positive")
    if not math.isfinite(outer_scale):
        raise ValueError("outer_scale must be finite")

    support = (mask > 0).to(dtype=chi.dtype)
    susceptibility = chi * support
    dilated = dilate_mask(support, dilation_radius).to(dtype=chi.dtype)
    exterior_source = 1.0 - dilated
    local_field = support * dipole_forward(susceptibility, dipole_kernel)
    outer_field = outer_scale * support * dipole_forward(
        exterior_source,
        dipole_kernel,
    )
    noise_std = float(chi.abs().max()) / float(snr)
    generator = torch.Generator(device=chi.device).manual_seed(seed)
    noise = support * noise_std * torch.randn(
        chi.shape,
        device=chi.device,
        dtype=chi.dtype,
        generator=generator,
    )
    return CosmosSimulation(
        field=local_field + outer_field + noise,
        local_field=local_field,
        outer_field=outer_field,
        noise=noise,
        dilated_mask=dilated,
        exterior_source=exterior_source,
        noise_std=noise_std,
        snr=float(snr),
        dilation_radius=dilation_radius,
        outer_scale=float(outer_scale),
    )
