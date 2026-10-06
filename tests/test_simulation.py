import torch

from wh_inr.operators import build_dipole_kernel, dipole_forward
from wh_inr.simulation import dilate_mask, simulate_cosmos, spherical_structuring_element


def test_radius_five_structuring_element_and_dilation() -> None:
    sphere = spherical_structuring_element(5)
    assert sphere.shape == (11, 11, 11)
    assert int(sphere.sum()) == 515
    mask = torch.zeros(15, 15, 15)
    mask[7, 7, 7] = 1
    dilated = dilate_mask(mask, radius=5)
    assert dilated[12, 7, 7] == 1
    assert dilated[11, 11, 7] == 0
    assert int(dilated.sum()) == 515


def test_cosmos_noise_and_outer_field_follow_requested_formulas() -> None:
    torch.manual_seed(12)
    shape = (12, 13, 14)
    mask = torch.zeros(shape)
    mask[2:-2, 2:-2, 2:-2] = 1
    chi = 0.2 * torch.randn(shape) * mask
    chi[0, 0, 0] = 0.7
    kernel = build_dipole_kernel(shape)
    result = simulate_cosmos(chi, mask, kernel, snr=80.0, seed=13)
    expected_sigma = float(chi.abs().max()) / 80.0
    expected_source = 1.0 - result.dilated_mask
    expected_outer = 10.0 * mask * dipole_forward(expected_source, kernel)

    assert result.noise_std == expected_sigma
    assert result.dilation_radius == 5
    assert result.outer_scale == 10.0
    torch.testing.assert_close(result.exterior_source, expected_source)
    torch.testing.assert_close(result.outer_field, expected_outer)
    torch.testing.assert_close(
        result.field,
        result.local_field + result.outer_field + result.noise,
    )
    assert torch.count_nonzero(result.noise[mask == 0]) == 0
