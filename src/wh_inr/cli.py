"""Command-line interface for simulation and WH-INR reconstruction."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from wh_inr.device import describe_device, resolve_device
from wh_inr.io import load_volume, save_arrays
from wh_inr.operators import build_dipole_kernel
from wh_inr.simulation import simulate_cosmos
from wh_inr.solver import ReconstructionConfig, reconstruct


def _add_geometry(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument(
        "--b0-direction",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("Z", "Y", "X"),
    )


def _simulate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    print(f"Using {describe_device(device).description()}")
    chi = load_volume(args.chi, ("chi", "chi_cosmos", "susceptibility"))
    mask = load_volume(args.mask, ("mask", "msk", "brain_mask"))
    if chi.shape != mask.shape:
        raise ValueError("chi and mask must have matching shapes")
    if bool((mask < 0).any()) or not bool((mask > 0).any()):
        raise ValueError("mask must be nonnegative and nonempty")
    if device.type == "cuda":
        chi = chi.pin_memory().to(device, non_blocking=True)
        mask = mask.pin_memory().to(device, non_blocking=True)
    else:
        chi = chi.to(device)
        mask = mask.to(device)
    kernel = build_dipole_kernel(
        chi.shape,
        args.voxel_size,
        args.b0_direction,
        device=device,
    )
    simulation = simulate_cosmos(
        chi,
        mask,
        kernel,
        snr=args.snr,
        seed=args.seed,
        dilation_radius=5,
        outer_scale=10.0,
    )
    masked_chi = chi * (mask > 0)
    save_arrays(
        args.output,
        {
            "b": simulation.field,
            "chi_reference": masked_chi,
            "local_field_reference": simulation.local_field,
            "h_reference": simulation.outer_field,
            "outer_field_reference": simulation.outer_field,
            "noise_reference": simulation.noise,
            "mask": mask,
            "Md": simulation.dilated_mask,
            "exterior_source": simulation.exterior_source,
            "dipole_kernel": kernel,
            "snr": torch.tensor(simulation.snr),
            "noise_std": torch.tensor(simulation.noise_std),
            "dilation_radius": torch.tensor(simulation.dilation_radius),
            "outer_scale": torch.tensor(simulation.outer_scale),
        },
    )
    print(
        f"Noise sigma=max(abs(chi))/SNR={simulation.noise_std:.6e} "
        f"for SNR={simulation.snr:g}"
    )
    print(f"Saved simulated field to {args.output}")


def _reconstruct(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    print(f"Using {describe_device(device).description()}")
    field = load_volume(args.field, ("b", "field", "phase", "local_field"))
    mask = load_volume(args.mask, ("mask", "msk", "brain_mask"))
    weight = (
        load_volume(args.weight, ("w", "weight", "magn", "magnitude"))
        if args.weight is not None
        else mask.clone()
    )
    if args.ground_truth is not None:
        ground_truth = load_volume(
            args.ground_truth,
            ("chi_reference", "chi_cosmos", "chi", "susceptibility", "ground_truth"),
        )
    elif args.field.suffix.lower() != ".npy":
        try:
            ground_truth = load_volume(
                args.field,
                ("chi_reference", "chi_cosmos", "chi", "susceptibility", "ground_truth"),
                allow_single=False,
            )
        except KeyError:
            ground_truth = None
    else:
        ground_truth = None
    if field.shape != mask.shape or field.shape != weight.shape:
        raise ValueError("field, mask, and weight must have matching shapes")
    if ground_truth is not None and ground_truth.shape != field.shape:
        raise ValueError("ground truth must have the same shape as field")
    if not args.weight_outside_mask:
        weight = weight * (mask > 0)
    config = ReconstructionConfig(
        beta=args.beta,
        tv_lambda=args.tv_lambda,
        outer_iterations=args.outer_iterations,
        siren_steps=args.siren_steps,
        learning_rate=args.learning_rate,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        omega_0=args.omega_0,
        hidden_omega=args.hidden_omega,
        chunk_size=args.chunk_size,
        cg_iterations=args.cg_iterations,
        cg_tolerance=args.cg_tolerance,
        outer_tolerance=args.outer_tolerance,
        zero_mean_chi=not args.keep_chi_mean,
        gradient_clip=args.gradient_clip,
        amp=args.amp,
        seed=args.seed,
    )

    def progress(entry: dict[str, float | int | bool]) -> None:
        memory = (
            f" gpu_peak={entry['cuda_peak_memory_allocated_mib']:.0f}MiB"
            if "cuda_peak_memory_allocated_mib" in entry
            else ""
        )
        nrmse = (
            f" nrmse={entry['nrmse_percent']:.3f}%"
            if "nrmse_percent" in entry
            else ""
        )
        cg = (
            f" cg_rel={entry['cg_relative_residual']:.2e}"
            if "cg_relative_residual" in entry
            else ""
        )
        print(
            f"outer={entry['outer_iteration']:>3} "
            f"J/|M|={entry['objective_per_mask_voxel']:.6e} "
            f"data={entry['data_term']:.6e} "
            f"harmonic={entry['harmonic_term']:.6e} "
            f"tv={entry['tv_term']:.6e}{nrmse}{cg}{memory}"
        )

    result = reconstruct(
        field,
        mask,
        weight,
        ground_truth,
        voxel_size=args.voxel_size,
        b0_direction=args.b0_direction,
        config=config,
        device=device,
        callback=progress,
    )
    arrays = {
        "chi": result.chi,
        "h": result.background,
        "predicted_field": result.predicted_field,
        "residual": result.predicted_field - field.to(result.predicted_field.device),
    }
    if result.ground_truth is not None:
        arrays.update(
            chi_ground_truth=result.ground_truth,
            chi_difference=result.chi - result.ground_truth,
        )
    save_arrays(args.output, arrays)
    checkpoint_path = args.checkpoint or Path(args.output).with_suffix(".pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.checkpoint(), checkpoint_path)
    history_path = Path(args.output).with_suffix(".json")
    metadata: dict[str, Any] = {
        "config": asdict(config),
        "voxel_size_zyx": args.voxel_size,
        "b0_direction_zyx": args.b0_direction,
        "device": str(device),
        "history": result.history,
    }
    if result.ground_truth is not None:
        metadata.update(
            nrmse={
                "definition": "||M (chi - GT)||_2 / ||M GT||_2",
                "reported_as": ["ratio", "percent"],
                "ground_truth_gauge": (
                    "masked mean removed" if config.zero_mean_chi else "as supplied"
                ),
            },
            visualization={
                "cmap": "gray",
                "susceptibility_limits": [-0.1, 0.1],
                "difference_limits": [-0.03, 0.03],
                "sections": ["central axial", "central coronal", "central sagittal"],
            },
        )
    history_path.write_text(json.dumps(metadata, indent=2) + "\n")
    from wh_inr.visualization import plot_input_field

    input_figure_path = args.input_figure or Path(args.output).with_name(
        f"{Path(args.output).stem}_input.png"
    )
    plot_input_field(
        field,
        mask,
        output=input_figure_path,
        show=False,
    )
    print(f"Saved input field plot to {input_figure_path}")
    if result.ground_truth is not None:
        from wh_inr.visualization import plot_nrmse_history, plot_orthogonal_comparison

        figure_path = args.figure or Path(args.output).with_name(
            f"{Path(args.output).stem}_comparison.png"
        )
        plot_orthogonal_comparison(
            result.chi,
            result.ground_truth,
            mask,
            output=figure_path,
            nrmse_percent=float(result.history[-1]["nrmse_percent"]),
            show=False,
        )
        nrmse_figure_path = args.nrmse_figure or Path(args.output).with_name(
            f"{Path(args.output).stem}_nrmse.png"
        )
        plot_nrmse_history(
            result.history,
            output=nrmse_figure_path,
            show=False,
        )
        print(f"Saved GT comparison to {figure_path}")
        print(f"Saved NRMSE history to {nrmse_figure_path}")
    else:
        print(
            "Ground truth was not found; pass --ground-truth to calculate "
            "NRMSE and create the comparison figure."
        )
    if args.show_figure:
        import matplotlib.pyplot as plt

        plt.show()
    print(f"Saved reconstruction to {args.output}")
    print(f"Saved SIREN checkpoint to {checkpoint_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wh-inr", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulation = subparsers.add_parser("simulate", help="simulate b from a reference chi")
    simulation.add_argument("--chi", type=Path, required=True)
    simulation.add_argument("--mask", type=Path, required=True)
    simulation.add_argument("--output", type=Path, required=True)
    simulation.add_argument("--snr", type=float, default=100.0)
    simulation.add_argument("--device", default="auto")
    simulation.add_argument("--seed", type=int, default=0)
    _add_geometry(simulation)
    simulation.set_defaults(handler=_simulate)

    reconstruction = subparsers.add_parser("reconstruct", help="estimate chi and h")
    reconstruction.add_argument("--field", type=Path, required=True)
    reconstruction.add_argument("--mask", type=Path, required=True)
    reconstruction.add_argument("--weight", type=Path)
    reconstruction.add_argument(
        "--ground-truth",
        type=Path,
        help="GT volume; defaults to chi_reference in the field file when present",
    )
    reconstruction.add_argument("--weight-outside-mask", action="store_true")
    reconstruction.add_argument("--output", type=Path, default=Path("outputs/reconstruction.npz"))
    reconstruction.add_argument("--checkpoint", type=Path)
    reconstruction.add_argument("--figure", type=Path)
    reconstruction.add_argument("--input-figure", type=Path)
    reconstruction.add_argument("--nrmse-figure", type=Path)
    reconstruction.add_argument("--show-figure", action="store_true")
    reconstruction.add_argument("--device", default="auto")
    reconstruction.add_argument("--beta", type=float, default=1500.0)
    reconstruction.add_argument(
        "--tv-lambda",
        type=float,
        default=1e-5,
        help="masked isotropic susceptibility-TV weight (default: 1e-5)",
    )
    reconstruction.add_argument("--outer-iterations", type=int, default=8)
    reconstruction.add_argument("--siren-steps", type=int, default=10)
    reconstruction.add_argument("--learning-rate", type=float, default=1e-4)
    reconstruction.add_argument("--hidden-features", type=int, default=128)
    reconstruction.add_argument("--hidden-layers", type=int, default=3)
    reconstruction.add_argument("--omega-0", type=float, default=30.0)
    reconstruction.add_argument("--hidden-omega", type=float, default=30.0)
    reconstruction.add_argument("--chunk-size", type=int, default=65_536)
    reconstruction.add_argument("--cg-iterations", type=int, default=500)
    reconstruction.add_argument("--cg-tolerance", type=float, default=1e-5)
    reconstruction.add_argument("--outer-tolerance", type=float, default=0.0)
    reconstruction.add_argument("--gradient-clip", type=float, default=1.0)
    reconstruction.add_argument(
        "--amp",
        action="store_true",
        help="use BF16 autocast for the SIREN MLP; FFTs and CG remain FP32",
    )
    reconstruction.add_argument("--keep-chi-mean", action="store_true")
    reconstruction.add_argument("--seed", type=int, default=0)
    _add_geometry(reconstruction)
    reconstruction.set_defaults(handler=_reconstruct)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
