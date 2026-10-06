"""Weak-harmonic implicit neural representation for QSM."""

from wh_inr.device import DeviceInfo, describe_device, resolve_device
from wh_inr.operators import build_dipole_kernel, dipole_forward, positive_laplacian
from wh_inr.siren import Siren
from wh_inr.simulation import CosmosSimulation, dilate_mask, simulate_cosmos
from wh_inr.solver import (
    ReconstructionConfig,
    ReconstructionResult,
    masked_total_variation,
    reconstruct,
    solve_background,
)

__all__ = [
    "ReconstructionConfig",
    "ReconstructionResult",
    "Siren",
    "DeviceInfo",
    "CosmosSimulation",
    "build_dipole_kernel",
    "describe_device",
    "dipole_forward",
    "dilate_mask",
    "positive_laplacian",
    "masked_total_variation",
    "reconstruct",
    "resolve_device",
    "simulate_cosmos",
    "solve_background",
]
