from pathlib import Path

import matplotlib.pyplot as plt
import torch

from wh_inr.visualization import (
    DIFFERENCE_LIMITS,
    SUSCEPTIBILITY_LIMITS,
    plot_input_field,
    plot_nrmse_history,
    plot_orthogonal_comparison,
)


def test_comparison_uses_requested_gray_ranges(tmp_path: Path) -> None:
    shape = (7, 8, 9)
    ground_truth = torch.linspace(-0.1, 0.1, 7 * 8 * 9).reshape(shape)
    prediction = ground_truth + 0.01
    mask = torch.ones(shape)
    output = tmp_path / "comparison.png"
    figure = plot_orthogonal_comparison(
        prediction,
        ground_truth,
        mask,
        output=output,
        nrmse_percent=10.0,
    )

    image_axes = figure.axes[:9]
    assert output.is_file() and output.stat().st_size > 0
    assert all(axis.images[0].get_cmap().name == "gray" for axis in image_axes)
    assert all(axis.images[0].get_clim() == SUSCEPTIBILITY_LIMITS for axis in image_axes[:6])
    assert all(axis.images[0].get_clim() == DIFFERENCE_LIMITS for axis in image_axes[6:])
    plt.close(figure)


def test_nrmse_history_plot_uses_iterations_and_percentages(tmp_path: Path) -> None:
    history: list[dict[str, float | int | bool]] = [
        {"outer_iteration": 0, "nrmse": 1.0, "nrmse_percent": 100.0},
        {"outer_iteration": 1, "nrmse": 0.75, "nrmse_percent": 75.0},
        {"outer_iteration": 2, "nrmse": 0.5, "nrmse_percent": 50.0},
    ]
    output = tmp_path / "nrmse.png"
    figure = plot_nrmse_history(history, output=output)
    axis = figure.axes[0]

    assert output.is_file() and output.stat().st_size > 0
    assert list(axis.lines[0].get_xdata()) == [0, 1, 2]
    assert list(axis.lines[0].get_ydata()) == [100.0, 75.0, 50.0]
    assert axis.get_xlabel() == "Outer iteration"
    assert axis.get_ylabel() == "Masked NRMSE (%)"
    plt.close(figure)


def test_nrmse_history_limits_tick_labels_for_long_runs() -> None:
    history: list[dict[str, float | int | bool]] = [
        {
            "outer_iteration": iteration,
            "nrmse": 1.0 / (iteration + 1),
            "nrmse_percent": 100.0 / (iteration + 1),
        }
        for iteration in range(101)
    ]
    figure = plot_nrmse_history(history)
    figure.canvas.draw()
    axis = figure.axes[0]
    visible_labels = [label for label in axis.get_xticklabels() if label.get_visible()]
    assert len(visible_labels) <= 12
    plt.close(figure)


def test_input_field_plot_has_three_gray_central_sections(tmp_path: Path) -> None:
    field = torch.linspace(-0.2, 0.2, 9 * 10 * 11).reshape(9, 10, 11)
    mask = torch.ones_like(field)
    output = tmp_path / "input.png"
    figure = plot_input_field(field, mask, output=output)

    image_axes = figure.axes[:3]
    assert output.is_file() and output.stat().st_size > 0
    assert len(image_axes) == 3
    assert all(axis.images[0].get_cmap().name == "gray" for axis in image_axes)
    assert all(
        axis.images[0].get_clim()[0] < 0 < axis.images[0].get_clim()[1]
        for axis in image_axes
    )
    plt.close(figure)
