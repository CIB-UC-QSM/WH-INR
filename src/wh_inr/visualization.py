"""Fixed-range orthogonal comparison figures for QSM reconstruction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from torch import Tensor

SUSCEPTIBILITY_LIMITS = (-0.1, 0.1)
DIFFERENCE_LIMITS = (-0.03, 0.03)


def plot_input_field(
    field: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
    *,
    output: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot masked central axial, coronal, and sagittal input-field sections."""

    field = _numpy_volume(field, "field")
    mask = _numpy_volume(mask, "mask") > 0
    if field.shape != mask.shape:
        raise ValueError("field and mask must have matching shapes")
    selected = np.abs(field[mask])
    if selected.size == 0:
        raise ValueError("mask must be nonempty")
    limit = float(np.quantile(selected, 0.995))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    display = np.where(mask, field, np.nan)
    orientations = ("Central axial", "Central coronal", "Central sagittal")
    colormap = plt.get_cmap("gray").with_extremes(bad="black")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    for axis, orientation, section in zip(
        axes,
        orientations,
        _central_sections(display),
        strict=True,
    ):
        image = axis.imshow(
            section,
            cmap=colormap,
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(orientation)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.8, label="Input field b")
    figure.suptitle(f"WH-INR input field | display range ±{limit:.4g}")

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def plot_nrmse_history(
    history: list[dict[str, float | int | bool]],
    *,
    output: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot masked NRMSE percentage against recorded outer iteration."""

    entries = [entry for entry in history if "nrmse_percent" in entry]
    if not entries:
        raise ValueError("history does not contain NRMSE values")
    iterations = [int(entry["outer_iteration"]) for entry in entries]
    values = [float(entry["nrmse_percent"]) for entry in entries]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    marker_interval = max(1, len(iterations) // 20)
    axis.plot(
        iterations,
        values,
        color="black",
        marker="o",
        markevery=marker_interval,
        linewidth=1.8,
    )
    axis.set_xlabel("Outer iteration")
    axis.set_ylabel("Masked NRMSE (%)")
    axis.set_title("WH-INR NRMSE versus iteration")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True, min_n_ticks=2))
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="0.85", linewidth=0.8)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def _numpy_volume(value: Tensor | np.ndarray, name: str) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value).squeeze()
    if result.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional volume")
    return result


def _central_sections(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = (size // 2 for size in volume.shape)
    return (
        np.rot90(volume[z, :, :]),
        np.rot90(volume[:, y, :]),
        np.rot90(volume[:, :, x]),
    )


def plot_orthogonal_comparison(
    prediction: Tensor | np.ndarray,
    ground_truth: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
    *,
    output: str | Path | None = None,
    nrmse_percent: float | None = None,
    show: bool = False,
) -> Figure:
    """Plot central axial, coronal, and sagittal GT/prediction/error sections."""

    prediction = _numpy_volume(prediction, "prediction")
    ground_truth = _numpy_volume(ground_truth, "ground_truth")
    mask = _numpy_volume(mask, "mask") > 0
    if prediction.shape != ground_truth.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, ground_truth, and mask must have matching shapes")

    difference = prediction - ground_truth
    rows = (
        ("Ground truth", np.where(mask, ground_truth, np.nan), SUSCEPTIBILITY_LIMITS),
        ("WH-INR prediction", np.where(mask, prediction, np.nan), SUSCEPTIBILITY_LIMITS),
        ("Prediction - GT", np.where(mask, difference, np.nan), DIFFERENCE_LIMITS),
    )
    orientations = ("Central axial", "Central coronal", "Central sagittal")
    colormap = plt.get_cmap("gray").with_extremes(bad="black")
    figure, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    row_images = []
    for row_index, (label, volume, limits) in enumerate(rows):
        sections = _central_sections(volume)
        for column_index, (orientation, section) in enumerate(
            zip(orientations, sections, strict=True)
        ):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                section,
                cmap=colormap,
                vmin=limits[0],
                vmax=limits[1],
                interpolation="nearest",
            )
            if row_index == 0:
                axis.set_title(orientation)
            if column_index == 0:
                axis.set_ylabel(label)
            axis.set_xticks([])
            axis.set_yticks([])
        row_images.append(image)
    for row_index, image in enumerate(row_images):
        label = "Susceptibility" if row_index < 2 else "Difference"
        figure.colorbar(image, ax=axes[row_index, :], shrink=0.75, label=label)
    title = "WH-INR central-section comparison"
    if nrmse_percent is not None:
        title += f" | masked NRMSE = {nrmse_percent:.3f}%"
    figure.suptitle(title)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure
