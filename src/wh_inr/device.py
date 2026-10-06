"""CUDA device selection and diagnostics."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    total_memory_mib: int | None
    compute_capability: tuple[int, int] | None

    def description(self) -> str:
        if self.device.type != "cuda":
            return str(self.device)
        major, minor = self.compute_capability or (0, 0)
        return (
            f"{self.device} ({self.name}, {self.total_memory_mib:,} MiB, "
            f"compute capability {major}.{minor})"
        )


def resolve_device(requested: torch.device | str | None = None) -> torch.device:
    """Resolve ``None``/``auto`` and validate an explicitly requested CUDA GPU."""

    if requested is None or str(requested).lower() == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        build = torch.version.cuda or "CPU-only"
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access an NVIDIA GPU "
            f"(PyTorch CUDA build: {build}). Check nvidia-smi, the driver, and "
            "container GPU access."
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {index} is unavailable; found "
            f"{torch.cuda.device_count()} device(s)"
        )
    return torch.device("cuda", index)


def describe_device(device: torch.device | str) -> DeviceInfo:
    """Return printable hardware information for a resolved device."""

    device = torch.device(device)
    if device.type != "cuda":
        return DeviceInfo(device, device.type.upper(), None, None)
    properties = torch.cuda.get_device_properties(device)
    return DeviceInfo(
        device=device,
        name=properties.name,
        total_memory_mib=round(properties.total_memory / 2**20),
        compute_capability=(properties.major, properties.minor),
    )

