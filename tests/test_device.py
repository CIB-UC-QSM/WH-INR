import pytest
import torch

from wh_inr.device import describe_device, resolve_device
from wh_inr.operators import build_dipole_kernel, dipole_forward
from wh_inr.solver import ReconstructionConfig, reconstruct


def test_cpu_device_resolution() -> None:
    device = resolve_device("cpu")
    assert device == torch.device("cpu")
    assert describe_device(device).description() == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
@pytest.mark.parametrize("amp", [False, True])
def test_cuda_reconstruction(amp: bool) -> None:
    device = resolve_device("cuda:0")
    shape = (8, 8, 8)
    mask = torch.ones(shape)
    kernel = build_dipole_kernel(shape)
    chi_reference = 0.05 * torch.randn(shape)
    field = dipole_forward(chi_reference, kernel)
    result = reconstruct(
        field,
        mask,
        config=ReconstructionConfig(
            beta=2.0,
            outer_iterations=1,
            siren_steps=1,
            hidden_features=8,
            hidden_layers=1,
            chunk_size=64,
            cg_iterations=4,
            amp=amp,
        ),
        device=device,
    )
    assert result.chi.device == device
    assert result.background.device == device
    assert torch.isfinite(result.chi).all()
    assert result.history[-1]["siren_amp"] is (
        amp and torch.cuda.is_bf16_supported()
    )
