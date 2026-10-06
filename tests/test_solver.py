import torch

from wh_inr.operators import build_dipole_kernel, dipole_forward, positive_laplacian
from wh_inr.siren import Siren, coordinate_grid
from wh_inr.solver import (
    ReconstructionConfig,
    functional_terms,
    masked_total_variation,
    reconstruct,
    solve_background,
)


def test_background_subproblem_reduces_its_objective() -> None:
    torch.manual_seed(4)
    shape = (7, 7, 7)
    mask = torch.zeros(shape)
    mask[1:-1, 1:-1, 1:-1] = 1
    weight = mask.clone()
    target = torch.randn(shape) * mask
    beta = 2.0
    solution, info = solve_background(
        target,
        weight,
        mask,
        beta=beta,
        max_iterations=300,
        tolerance=1e-6,
    )

    def objective(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum((weight * (value - target)).square()) + 0.5 * beta * torch.sum(
            (mask * positive_laplacian(value)).square()
        )

    assert objective(solution) < objective(torch.zeros_like(solution))
    assert info.relative_residual < 2e-5


def test_tiny_alternating_reconstruction_runs_and_returns_finite_state() -> None:
    torch.manual_seed(5)
    shape = (6, 6, 6)
    mask = torch.ones(shape)
    kernel = build_dipole_kernel(shape)
    chi_reference = 0.05 * torch.randn(shape)
    field = dipole_forward(chi_reference, kernel)
    config = ReconstructionConfig(
        beta=2.0,
        outer_iterations=2,
        siren_steps=2,
        learning_rate=5e-5,
        hidden_features=12,
        hidden_layers=1,
        chunk_size=64,
        cg_iterations=40,
        cg_tolerance=1e-5,
        seed=6,
    )
    result = reconstruct(
        field,
        mask,
        ground_truth=chi_reference,
        config=config,
        device="cpu",
    )
    assert result.chi.shape == shape
    assert result.background.shape == shape
    assert len(result.history) == 3
    assert torch.isfinite(result.chi).all()
    assert torch.isfinite(result.background).all()
    assert result.ground_truth is not None
    assert all("nrmse" in entry and "nrmse_percent" in entry for entry in result.history)
    assert all(float(entry["nrmse"]) >= 0.0 for entry in result.history)
    assert result.history[0]["nrmse_percent"] == 100.0
    assert "cg_relative_residual" in result.history[0]
    torch.testing.assert_close(
        result.ground_truth[mask > 0].mean(),
        torch.tensor(0.0),
        atol=1e-7,
        rtol=0.0,
    )
    data, harmonic, tv = functional_terms(
        result.chi,
        result.background,
        field,
        mask,
        mask,
        result.dipole_kernel,
        config.beta,
        config.tv_lambda,
    )
    assert torch.isfinite(data + harmonic + tv)
    assert all("tv_term" in entry for entry in result.history)


def test_masked_total_variation_ignores_outside_mask_and_constants() -> None:
    mask = torch.zeros(5, 5, 5)
    mask[1:-1, 1:-1, 1:-1] = 1
    constant = 0.25 * mask
    outside_change = constant.clone()
    outside_change[0, 0, 0] = 100.0

    torch.testing.assert_close(masked_total_variation(constant, mask), torch.tensor(0.0))
    torch.testing.assert_close(
        masked_total_variation(outside_change, mask),
        torch.tensor(0.0),
    )
    impulse = constant.clone()
    impulse[2, 2, 2] += 1.0
    assert masked_total_variation(impulse, mask) > 0.0


def test_chunked_siren_gradient_matches_full_volume_autograd() -> None:
    torch.manual_seed(7)
    shape = (4, 4, 4)
    mask = torch.ones(shape, dtype=torch.bool)
    mask[0, 0, 0] = False
    active = mask.reshape(-1)
    coordinates = coordinate_grid(
        shape,
        (1.0, 1.0, 1.0),
        device="cpu",
        dtype=torch.float32,
    )[active]
    kernel = build_dipole_kernel(shape)
    field = torch.randn(shape)
    full_model = Siren(hidden_features=8, hidden_layers=1)
    chunked_model = Siren(hidden_features=8, hidden_layers=1)
    chunked_model.load_state_dict(full_model.state_dict())

    raw = full_model(coordinates).squeeze(-1)
    chi = torch.zeros(active.numel())
    chi[active] = raw - raw.mean()
    residual = dipole_forward(chi.reshape(shape), kernel) - field
    tv_lambda = 3e-2
    (
        (
            0.5 * residual.square().sum()
            + tv_lambda * masked_total_variation(chi.reshape(shape), mask.reshape(shape))
        )
        / active.sum()
    ).backward()

    with torch.no_grad():
        raw = chunked_model(coordinates).squeeze(-1)
        chi = torch.zeros(active.numel())
        chi[active] = raw - raw.mean()
        residual = dipole_forward(chi.reshape(shape), kernel) - field
        data_gradient = dipole_forward(residual, kernel)
    with torch.enable_grad():
        differentiable_chi = chi.detach().reshape(shape).requires_grad_(True)
        tv_value = masked_total_variation(differentiable_chi, mask.reshape(shape))
        tv_gradient = torch.autograd.grad(tv_value, differentiable_chi)[0]
    with torch.no_grad():
        gradient = (data_gradient + tv_lambda * tv_gradient).reshape(-1)[active]
        gradient = gradient / active.sum()
        gradient -= gradient.mean()
    for start in range(0, coordinates.shape[0], 5):
        prediction = chunked_model(coordinates[start : start + 5]).squeeze(-1)
        prediction.backward(gradient[start : start + 5])

    for full_parameter, chunked_parameter in zip(
        full_model.parameters(), chunked_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            full_parameter.grad,
            chunked_parameter.grad,
            rtol=2e-5,
            atol=2e-6,
        )
