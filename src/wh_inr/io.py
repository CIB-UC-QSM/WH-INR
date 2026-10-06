"""Small, explicit volume I/O helpers for the command-line interface."""

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import scipy.io
import torch
from torch import Tensor


def load_volume(
    path: str | Path,
    keys: Sequence[str],
    *,
    allow_single: bool = True,
) -> Tensor:
    """Load a real 3D array from MAT, NPZ, or NPY."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        values = scipy.io.loadmat(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            values = {key: archive[key] for key in archive.files}
    elif suffix == ".npy":
        values = {keys[0]: np.load(path)}
    else:
        raise ValueError(f"unsupported volume format: {path.suffix}")
    key = next((candidate for candidate in keys if candidate in values), None)
    if key is None:
        visible = sorted(name for name in values if not name.startswith("__"))
        if allow_single and len(visible) == 1:
            key = visible[0]
        else:
            raise KeyError(f"none of {tuple(keys)} found in {path}; available keys: {visible}")
    array = np.asarray(values[key]).squeeze()
    if array.ndim != 3 or np.iscomplexobj(array) or not np.isfinite(array).all():
        raise ValueError(f"{path}:{key} must be a finite real 3D array")
    return torch.from_numpy(array.astype(np.float32, copy=False))


def save_arrays(path: str | Path, arrays: dict[str, Tensor | np.ndarray]) -> None:
    """Save named arrays to MAT or compressed NPZ."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = {
        key: value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
        for key, value in arrays.items()
    }
    if path.suffix.lower() == ".mat":
        scipy.io.savemat(path, converted, do_compression=True)
    elif path.suffix.lower() == ".npz":
        np.savez_compressed(path, **converted)
    else:
        raise ValueError("output path must end in .mat or .npz")
