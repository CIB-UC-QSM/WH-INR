"""Alternating weak-harmonic SIREN reconstruction."""

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor

from wh_inr.device import resolve_device
from wh_inr.operators import build_dipole_kernel, dipole_forward, positive_laplacian
from wh_inr.siren import Siren, coordinate_grid


@dataclass(frozen=True)
class ReconstructionConfig:
    """Numerical parameters for alternating reconstruction."""

    beta: float = 1500.0
    tv_lambda: float = 1e-5
    outer_iterations: int = 8
    siren_steps: int = 10
    learning_rate: float = 1e-4
    hidden_features: int = 128
    hidden_layers: int = 3
    omega_0: float = 30.0
    hidden_omega: float = 30.0
    chunk_size: int = 65_536
    cg_iterations: int = 500
    cg_tolerance: float = 1e-5
    outer_tolerance: float = 0.0
    zero_mean_chi: bool = True
    gradient_clip: float | None = 1.0
    amp: bool = False
    seed: int = 0

    def validate(self) -> None:
        positive_float = {
            "beta": self.beta,
            "learning_rate": self.learning_rate,
            "omega_0": self.omega_0,
            "hidden_omega": self.hidden_omega,
            "cg_tolerance": self.cg_tolerance,
        }
        for name, value in positive_float.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative_float = {"outer_tolerance": self.outer_tolerance}
        nonnegative_float["tv_lambda"] = self.tv_lambda
        for name, value in nonnegative_float.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        positive_int = {
            "outer_iterations": self.outer_iterations,
            "siren_steps": self.siren_steps,
            "hidden_features": self.hidden_features,
            "chunk_size": self.chunk_size,
            "cg_iterations": self.cg_iterations,
        }
        for name, value in positive_int.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_layers < 0:
            raise ValueError("hidden_layers must be nonnegative")
        if self.gradient_clip is not None and (
            not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0.0
        ):
            raise ValueError("gradient_clip must be finite and positive, or None")


@dataclass(frozen=True)
class CGInfo:
    iterations: int
    relative_residual: float
    converged: bool


@dataclass
class ReconstructionResult:
    chi: Tensor
    background: Tensor
    predicted_field: Tensor
    history: list[dict[str, float | int | bool]]
    model: Siren
    dipole_kernel: Tensor
    config: ReconstructionConfig
    voxel_size: tuple[float, float, float]
    b0_direction: tuple[float, float, float]
    ground_truth: Tensor | None

    def checkpoint(self) -> dict[str, object]:
        return {
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
            },
            "shape": tuple(self.chi.shape),
            "config": asdict(self.config),
            "voxel_size_zyx": self.voxel_size,
            "b0_direction_zyx": self.b0_direction,
        }


def _validate_volume(name: str, value: Tensor, shape: tuple[int, int, int] | None = None) -> Tensor:
    value = torch.as_tensor(value)
    if value.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional volume")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be real and finite")
    return value


@torch.no_grad()
def _conjugate_gradient(
    operator: Callable[[Tensor], Tensor],
    rhs: Tensor,
    initial: Tensor,
    preconditioner: Callable[[Tensor], Tensor],
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[Tensor, CGInfo]:
    solution = initial.clone()
    residual = rhs - operator(solution)
    rhs_norm = torch.linalg.vector_norm(rhs)
    scale = max(float(rhs_norm), torch.finfo(rhs.dtype).tiny)
    relative = float(torch.linalg.vector_norm(residual)) / scale
    if relative <= tolerance:
        return solution, CGInfo(0, relative, True)
    preconditioned = preconditioner(residual)
    direction = preconditioned.clone()
    residual_product = torch.sum(residual * preconditioned)
    tiny = torch.finfo(rhs.dtype).tiny

    for iteration in range(1, max_iterations + 1):
        applied = operator(direction)
        curvature = torch.sum(direction * applied)
        if not bool(torch.isfinite(curvature)) or float(curvature) <= tiny:
            return solution, CGInfo(iteration - 1, relative, False)
        step = residual_product / curvature
        solution.add_(direction, alpha=float(step))
        residual.sub_(applied, alpha=float(step))
        relative = float(torch.linalg.vector_norm(residual)) / scale
        if not math.isfinite(relative):
            raise FloatingPointError("nonfinite conjugate-gradient residual")
        if relative <= tolerance:
            return solution, CGInfo(iteration, relative, True)
        preconditioned = preconditioner(residual)
        next_product = torch.sum(residual * preconditioned)
        coefficient = next_product / residual_product
        direction.mul_(coefficient).add_(preconditioned)
        residual_product = next_product
    return solution, CGInfo(max_iterations, relative, False)


@torch.no_grad()
def solve_background(
    target: Tensor,
    weight: Tensor,
    mask: Tensor,
    *,
    beta: float,
    initial: Tensor | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-5,
) -> tuple[Tensor, CGInfo]:
    """Minimize the functional over ``h`` for a fixed dipole field.

    ``target`` is ``b - F^H D F chi``. The normal equation is
    ``(W^2 + beta L M^2 L) h = W^2 target``, where ``L=-Delta``.
    """

    target = _validate_volume("target", target)
    shape = tuple(target.shape)
    weight = _validate_volume("weight", weight, shape).to(target)
    mask = _validate_volume("mask", mask, shape).to(target)
    if bool((weight < 0).any()):
        raise ValueError("weight must be nonnegative")
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    squared_weight = weight.square()
    squared_mask = mask.square()

    def normal_operator(value: Tensor) -> Tensor:
        return squared_weight * value + beta * positive_laplacian(
            squared_mask * positive_laplacian(value)
        )

    # The dominant term is a masked biharmonic operator. Its periodic,
    # constant-coefficient approximation is diagonal in Fourier space and is
    # a substantially better preconditioner than a voxelwise Jacobi diagonal.
    laplacian_spectrum = torch.zeros_like(target)
    for axis, size in enumerate(shape):
        frequency = torch.fft.fftfreq(
            size,
            device=target.device,
            dtype=target.dtype,
        )
        axis_spectrum = 4.0 * torch.sin(math.pi * frequency).square()
        view_shape = [1, 1, 1]
        view_shape[axis] = size
        laplacian_spectrum += axis_spectrum.reshape(view_shape)
    preconditioner_spectrum = (
        squared_weight.mean()
        + beta * squared_mask.mean() * laplacian_spectrum.square()
    )

    def precondition(value: Tensor) -> Tensor:
        return torch.fft.ifftn(
            torch.fft.fftn(value, dim=(-3, -2, -1)) / preconditioner_spectrum,
            dim=(-3, -2, -1),
        ).real

    rhs = squared_weight * target
    initial = torch.zeros_like(target) if initial is None else _validate_volume(
        "initial", initial, shape
    ).to(target)
    return _conjugate_gradient(
        normal_operator,
        rhs,
        initial,
        precondition,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


@torch.no_grad()
def _evaluate_siren(
    model: Siren,
    coordinates: Tensor,
    active: Tensor,
    shape: tuple[int, int, int],
    chunk_size: int,
    zero_mean: bool,
    amp: bool,
) -> Tensor:
    values = torch.empty(coordinates.shape[0], device=coordinates.device, dtype=coordinates.dtype)
    model.eval()
    for start in range(0, coordinates.shape[0], chunk_size):
        stop = min(start + chunk_size, coordinates.shape[0])
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp
            else nullcontext()
        )
        with context:
            prediction = model(coordinates[start:stop]).squeeze(-1)
        values[start:stop] = prediction.float()
    if zero_mean:
        values -= values.mean()
    flat = torch.zeros(active.numel(), device=coordinates.device, dtype=coordinates.dtype)
    flat[active] = values
    return flat.reshape(shape)


def masked_total_variation(
    chi: Tensor,
    mask: Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Return differentiable isotropic TV over in-mask neighbour pairs."""

    if chi.ndim != 3 or mask.shape != chi.shape:
        raise ValueError("chi and mask must be matching three-dimensional volumes")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    support = (mask > 0).to(dtype=chi.dtype)
    squared_magnitude = torch.zeros_like(chi)
    for axis in (-3, -2, -1):
        edge = support * torch.roll(support, shifts=-1, dims=axis)
        difference = edge * (torch.roll(chi, shifts=-1, dims=axis) - chi)
        squared_magnitude = squared_magnitude + difference.square()
    return torch.sum(
        support * (torch.sqrt(squared_magnitude + epsilon**2) - epsilon)
    )


def functional_terms(
    chi: Tensor,
    background: Tensor,
    field: Tensor,
    weight: Tensor,
    mask: Tensor,
    kernel: Tensor,
    beta: float,
    tv_lambda: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the three unnormalized terms of the requested functional."""

    residual = dipole_forward(chi, kernel) + background - field
    data_term = 0.5 * torch.sum((weight * residual).square())
    harmonic_term = 0.5 * beta * torch.sum(
        (mask * positive_laplacian(background)).square()
    )
    tv_term = tv_lambda * masked_total_variation(chi, mask)
    return data_term, harmonic_term, tv_term


def reconstruct(
    field: Tensor,
    mask: Tensor,
    weight: Tensor | None = None,
    ground_truth: Tensor | None = None,
    *,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    b0_direction: Sequence[float] = (0.0, 0.0, 1.0),
    config: ReconstructionConfig | None = None,
    device: torch.device | str | None = None,
    callback: Callable[[dict[str, float | int | bool]], None] | None = None,
) -> ReconstructionResult:
    """Estimate ``chi`` and ``h`` by alternating SIREN and linear updates.

    The SIREN gradient is accumulated in coordinate chunks. This computes the
    same full-volume gradient as ordinary backpropagation but avoids retaining
    all coordinate activations at once.
    """

    config = ReconstructionConfig() if config is None else config
    config.validate()
    voxel_size = tuple(float(value) for value in voxel_size)
    b0_direction = tuple(float(value) for value in b0_direction)
    selected_device = resolve_device(device)

    def transfer(value: Tensor) -> Tensor:
        non_blocking = selected_device.type == "cuda" and value.device.type == "cpu"
        if non_blocking and not value.is_pinned():
            value = value.pin_memory()
        return value.to(
            device=selected_device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )

    field = transfer(_validate_volume("field", field))
    shape = tuple(field.shape)
    mask = transfer(_validate_volume("mask", mask, shape))
    if bool((mask < 0).any()) or not bool((mask > 0).any()):
        raise ValueError("mask must be nonnegative and nonempty")
    weight = mask if weight is None else transfer(_validate_volume("weight", weight, shape))
    if bool((weight < 0).any()) or not bool((weight > 0).any()):
        raise ValueError("weight must be nonnegative and nonzero")
    torch.manual_seed(config.seed)
    if selected_device.type == "cuda":
        torch.cuda.set_device(selected_device)
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(selected_device)
    use_amp = config.amp and selected_device.type == "cuda" and torch.cuda.is_bf16_supported()

    kernel = build_dipole_kernel(
        shape,
        voxel_size,
        b0_direction,
        device=selected_device,
        dtype=field.dtype,
    )
    model = Siren(
        hidden_features=config.hidden_features,
        hidden_layers=config.hidden_layers,
        omega_0=config.omega_0,
        hidden_omega=config.hidden_omega,
    ).to(device=selected_device, dtype=field.dtype)
    # A random initial susceptibility makes the first background solve absorb
    # arbitrary INR structure. More critically, updating chi while h is zero
    # makes the INR invert the much larger background field. Start from the
    # neutral chi=0 state and estimate h before the first chi update.
    with torch.no_grad():
        final_layer = model.network[-1]
        final_layer.weight.zero_()
        final_layer.bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    active = (mask.reshape(-1) > 0)
    referenced_ground_truth: Tensor | None = None
    if ground_truth is not None:
        referenced_ground_truth = transfer(
            _validate_volume("ground_truth", ground_truth, shape)
        )
        referenced_ground_truth = torch.where(
            active.reshape(shape),
            referenced_ground_truth,
            torch.zeros_like(referenced_ground_truth),
        )
        if config.zero_mean_chi:
            referenced_ground_truth = referenced_ground_truth.clone()
            referenced_ground_truth.reshape(-1)[active] -= referenced_ground_truth.reshape(-1)[
                active
            ].mean()
        if float(torch.linalg.vector_norm(referenced_ground_truth.reshape(-1)[active])) == 0.0:
            raise ValueError("ground_truth must be nonzero inside the mask")
    all_coordinates = coordinate_grid(
        shape,
        voxel_size,
        device=selected_device,
        dtype=field.dtype,
    )
    coordinates = all_coordinates[active]
    del all_coordinates
    normalizer = float(active.sum())
    background = torch.zeros_like(field)
    history: list[dict[str, float | int | bool]] = []

    def record(outer: int, chi: Tensor, cg: CGInfo | None) -> dict[str, float | int | bool]:
        with torch.no_grad():
            data_term, harmonic_term, tv_term = functional_terms(
                chi,
                background,
                field,
                weight,
                mask,
                kernel,
                config.beta,
                config.tv_lambda,
            )
            predicted = dipole_forward(chi, kernel) + background
            weighted_residual = weight * (predicted - field)
            entry: dict[str, float | int | bool] = {
                "outer_iteration": outer,
                "objective": float(data_term + harmonic_term + tv_term),
                "objective_per_mask_voxel": float(
                    (data_term + harmonic_term + tv_term) / normalizer
                ),
                "data_term": float(data_term),
                "harmonic_term": float(harmonic_term),
                "tv_term": float(tv_term),
                "weighted_residual_rms": float(weighted_residual.square().mean().sqrt()),
            }
            if referenced_ground_truth is not None:
                error = (chi - referenced_ground_truth).reshape(-1)[active]
                reference = referenced_ground_truth.reshape(-1)[active]
                nrmse = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)
                entry.update(
                    nrmse=float(nrmse),
                    nrmse_percent=100.0 * float(nrmse),
                )
            if cg is not None:
                entry.update(
                    cg_iterations=cg.iterations,
                    cg_relative_residual=cg.relative_residual,
                    cg_converged=cg.converged,
                )
            if selected_device.type == "cuda":
                entry.update(
                    cuda_memory_allocated_mib=round(
                        torch.cuda.memory_allocated(selected_device) / 2**20,
                        2,
                    ),
                    cuda_peak_memory_allocated_mib=round(
                        torch.cuda.max_memory_allocated(selected_device) / 2**20,
                        2,
                    ),
                    siren_amp=use_amp,
                )
        history.append(entry)
        if callback is not None:
            callback(entry)
        return entry

    chi = _evaluate_siren(
        model,
        coordinates,
        active,
        shape,
        config.chunk_size,
        config.zero_mean_chi,
        use_amp,
    )
    initial_local_field = dipole_forward(chi, kernel)
    background, initial_cg_info = solve_background(
        field - initial_local_field,
        weight,
        mask,
        beta=config.beta,
        initial=background,
        max_iterations=config.cg_iterations,
        tolerance=config.cg_tolerance,
    )
    record(0, chi, initial_cg_info)

    for outer in range(1, config.outer_iterations + 1):
        previous_chi = chi
        for _ in range(config.siren_steps):
            chi = _evaluate_siren(
                model,
                coordinates,
                active,
                shape,
                config.chunk_size,
                config.zero_mean_chi,
                use_amp,
            )
            if config.tv_lambda > 0.0:
                with torch.enable_grad():
                    differentiable_chi = chi.detach().requires_grad_(True)
                    tv_value = masked_total_variation(differentiable_chi, mask)
                    tv_gradient = torch.autograd.grad(tv_value, differentiable_chi)[0]
            else:
                tv_gradient = torch.zeros_like(chi)
            with torch.no_grad():
                residual = dipole_forward(chi, kernel) + background - field
                data_gradient = dipole_forward(weight.square() * residual, kernel)
                chi_gradient = (
                    data_gradient + config.tv_lambda * tv_gradient
                ) / normalizer
                active_gradient = chi_gradient.reshape(-1)[active]
                if config.zero_mean_chi:
                    active_gradient = active_gradient - active_gradient.mean()
            optimizer.zero_grad(set_to_none=True)
            model.train()
            for start in range(0, coordinates.shape[0], config.chunk_size):
                stop = min(start + config.chunk_size, coordinates.shape[0])
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_amp
                    else nullcontext()
                )
                with context:
                    prediction = model(coordinates[start:stop]).squeeze(-1)
                prediction.backward(active_gradient[start:stop])
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

        chi = _evaluate_siren(
            model,
            coordinates,
            active,
            shape,
            config.chunk_size,
            config.zero_mean_chi,
            use_amp,
        )
        local_field = dipole_forward(chi, kernel)
        background, cg_info = solve_background(
            field - local_field,
            weight,
            mask,
            beta=config.beta,
            initial=background,
            max_iterations=config.cg_iterations,
            tolerance=config.cg_tolerance,
        )
        record(outer, chi, cg_info)
        if config.outer_tolerance > 0.0:
            denominator = torch.linalg.vector_norm(previous_chi).clamp_min(
                torch.finfo(chi.dtype).tiny
            )
            relative_update = float(torch.linalg.vector_norm(chi - previous_chi) / denominator)
            history[-1]["chi_relative_update"] = relative_update
            if relative_update <= config.outer_tolerance:
                break

    predicted_field = dipole_forward(chi, kernel) + background
    return ReconstructionResult(
        chi=chi.detach(),
        background=background.detach(),
        predicted_field=predicted_field.detach(),
        history=history,
        model=model,
        dipole_kernel=kernel.detach(),
        config=config,
        voxel_size=voxel_size,  # type: ignore[arg-type]
        b0_direction=b0_direction,  # type: ignore[arg-type]
        ground_truth=(
            None if referenced_ground_truth is None else referenced_ground_truth.detach()
        ),
    )
