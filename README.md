# WH-INR

WH-INR estimates a quantitative susceptibility map as a SIREN implicit neural
representation while separating a weak-harmonic background field. It minimizes

\[
J(\chi,h)=\frac{1}{2}\left\|w\left(F^HDF\chi+h-b\right)\right\|_2^2
+\frac{\beta}{2}\left\|M\Delta h\right\|_2^2
+\lambda\,\operatorname{TV}_M(\chi),
\]

where `b` is the measured field, `D` is the k-space dipole kernel, `w` is the
data weight, and `M` is the spatial mask. The susceptibility prior is isotropic
TV over forward-difference edges whose two voxels are inside `M`.

## Method

The solver alternates between the two variables:

1. With `h` fixed, evaluate the whole SIREN susceptibility volume and optimize
   its parameters using the exact gradient of the data term.
2. With `chi` fixed, solve

   \[
   \left(W^2+\beta L M^2 L\right)h
   =W^2\left(b-F^HDF\chi\right),\qquad L=-\Delta,
   \]

   by preconditioned conjugate gradients.
3. Repeat the two updates for the configured number of outer iterations.

The SIREN gradient is accumulated over coordinate chunks. The Fourier forward
model still uses the complete volume, as required by the nonlocal dipole
operator, but neural activations for all coordinates are never kept in memory
at the same time. Susceptibility is zero outside the mask and, by default, its
masked mean is removed to fix the dipole operator's constant-offset ambiguity.

Both the dipole operator and finite-difference Laplacian use periodic boundary
conditions. Array axes, voxel sizes, and the B0 direction are all specified in
`(z, y, x)` order.

### Numerical stability

The susceptibility output layer is initialized to zero, followed by a
background solve before the first SIREN step. This ordering is important when
the exterior field is much larger than the local susceptibility field. The
background normal equation uses a Fourier biharmonic preconditioner; a
voxelwise diagonal preconditioner converges too slowly for this masked problem
and can leave a structured residual that is amplified by dipole inversion.

The simulated noise is homoscedastic. Use the mask as `w` (omit `--weight`) for
a statistically matched reconstruction. Supply a magnitude weight only when it
represents the inverse spatial noise standard deviation. Since multiplying `w`
changes the relative scale of both functional terms, retune `beta` whenever the
weight normalization changes.

## Installation

The project is locked for Python 3.13 and managed with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.13 --all-groups
uv run pytest
```

PyTorch automatically uses `cuda:0` when CUDA is available. Pass `--device cpu`
or a specific GPU such as `--device cuda:0` to override this choice. At startup,
the CLI prints the selected GPU, VRAM, and compute capability.

The Fourier dipole model, objective, and conjugate-gradient background solve
always run in FP32. On GPUs with BF16 support, `--amp` enables BF16 autocast for
the SIREN MLP only. This reduces neural activation memory while keeping the
physics and linear solve in FP32:

```bash
uv run wh-inr reconstruct \
  --device cuda:0 \
  --amp \
  --field outputs/simulated.mat \
  --mask msk.mat \
  --weight magn.mat \
  --ground-truth chi_cosmos.mat \
  --output outputs/reconstruction.npz
```

Input volumes are transferred through pinned host memory. Iteration history
records current and peak CUDA allocation in MiB. Reduce `--chunk-size` if a
larger SIREN or another process leaves insufficient GPU memory.

## Run the included COSMOS example

The repository's `chi_cosmos.mat`, `magn.mat`, and `msk.mat` files have matching
`160 x 160 x 160` volumes. Since they do not include a measured field `b`, first
create one with the COSMOS exterior-field and peak-SNR simulation:

```bash
uv run wh-inr simulate \
  --device cuda:0 \
  --chi chi_cosmos.mat \
  --mask msk.mat \
  --snr 100 \
  --output outputs/simulated.mat
```

The simulator constructs a radius-5 spherical dilation `Md` of the binary mask
and uses

\[
h_{\mathrm{out}}=10M F^H D F (1-M_d),\qquad
\sigma_{\mathrm{noise}}=\frac{\max|\chi|}{\mathrm{SNR}}.
\]

It adds masked Gaussian noise with this standard deviation to the masked local
and outer fields. The MAT output includes `Md`, `exterior_source=1-Md`,
`local_field_reference`, `outer_field_reference`, `noise_reference`, `snr`, and
`noise_std` so every simulation component can be inspected independently.

Then reconstruct `chi` and `h`:

```bash
uv run wh-inr reconstruct \
  --field outputs/simulated.mat \
  --mask msk.mat \
  --weight magn.mat \
  --ground-truth chi_cosmos.mat \
  --beta 1500 \
  --tv-lambda 1e-5 \
  --outer-iterations 8 \
  --siren-steps 10 \
  --output outputs/reconstruction.npz
```

The simulator uses the reference susceptibility to form `b`. During
reconstruction, GT is used only for NRMSE and visualization, never as an
optimization input.
The SIREN begins from `chi=0`, and the reconstruction solves an initial
weak-harmonic background before the first susceptibility update. This prevents
the much larger exterior field from being encoded into the initial
susceptibility estimate. The background solve uses a Fourier biharmonic
preconditioner for the poorly conditioned masked system; the default budget is
500 preconditioned CG iterations.
When the field file contains `chi_reference`, it is detected automatically, so
the explicit `--ground-truth` argument can be omitted. By default the command
line uses
`w = supplied_weight * mask`, which is the usual convention when the field is
valid only inside the brain. Add `--weight-outside-mask` when `b` and `w` are
defined on the complete domain.

The reconstruction command writes:

- `reconstruction.npz`: `chi`, `h`, the fitted field, the field residual, the
  gauge-aligned GT, and `chi - GT`;
- `reconstruction.pt`: the trained SIREN parameters and configuration;
- `reconstruction.json`: the objective components and CG diagnostics after
  every outer iteration, including masked NRMSE;
- `reconstruction_comparison.png`: central axial, coronal, and sagittal GT,
  prediction, and difference sections;
- `reconstruction_input.png`: masked central axial, coronal, and sagittal
  sections of the input field `b`;
- `reconstruction_nrmse.png`: masked NRMSE percentage versus outer iteration.

The NRMSE graph uses at most about ten integer iteration labels and reduces the
number of point markers for long runs, keeping the x-axis readable when many
outer iterations are requested.

For every recorded outer iteration, the script calculates

\[
\operatorname{NRMSE}(\chi_k,\chi_{GT}) =
\frac{\left\|M(\chi_k-\chi_{GT})\right\|_2}
{\left\|M\chi_{GT}\right\|_2}.
\]

It prints the percentage and stores both the ratio and percentage in JSON. The
comparison figure uses `cmap="gray"`. GT and prediction panels have fixed limits
`[-0.1, 0.1]`; difference panels have fixed limits `[-0.03, 0.03]`. Values
outside these display limits saturate at the endpoints without altering the
saved numerical arrays. When zero-mean susceptibility referencing is active,
the masked GT mean is removed before NRMSE and visualization so it uses the same
dipole-model gauge as the reconstruction.

MAT files may use `b`, `field`, `phase`, or `local_field` for the measured
field; `mask`, `msk`, or `brain_mask` for the mask; `w`, `weight`, `magn`, or
`magnitude` for weights; and `chi_reference`, `chi_cosmos`, `chi`,
`susceptibility`, or `ground_truth` for GT. NPY and NPZ inputs are also
supported. Outputs can be MAT or NPZ files.

## Python API

```python
import torch

from wh_inr import ReconstructionConfig, reconstruct

config = ReconstructionConfig(
    beta=1500.0,
    tv_lambda=1e-5,
    outer_iterations=8,
    siren_steps=10,
    hidden_features=128,
)

result = reconstruct(
    field=b,
    mask=mask,
    weight=w,
    ground_truth=chi_gt,
    voxel_size=(1.0, 1.0, 1.0),
    b0_direction=(0.0, 0.0, 1.0),
    config=config,
)

chi = result.chi
h = result.background
nrmse_percent = [entry["nrmse_percent"] for entry in result.history]
```

The Python API uses `w` exactly as supplied. The CLI mask multiplication
described above is a data-loading convention rather than part of the objective.

## Main parameters

- `beta` controls how strongly the background is driven toward harmonicity.
- `tv_lambda` controls susceptibility TV regularization and defaults to `1e-5`.
- `siren_steps` controls the number of fixed-background SIREN updates per
  outer iteration.
- `cg_iterations` and `cg_tolerance` control the fixed-susceptibility solve.
- `chunk_size` trades GPU memory for SIREN evaluation speed.
- `omega_0` and `hidden_omega` set the SIREN frequency scales.

The default settings are starting values rather than acquisition-independent
QSM parameters. Weight scaling, field units, voxel size, B0 orientation, and
the chosen `beta` must be consistent for a physical reconstruction.

For the 96 GB RTX PRO 6000, `--chunk-size 262144` is a practical starting
point. The portable default is 65,536 coordinates for compatibility with
smaller GPUs.
