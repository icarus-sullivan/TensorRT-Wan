"""GPU detection.

Detection is deliberately lazy/optional at import time: this module must import cleanly on a
machine with no NVIDIA GPU and no CUDA-enabled torch build (development laptops, CI), and only
raise when a caller actually asks for GPU info that isn't there.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GPUArchitecture(str, Enum):
    """NVIDIA microarchitecture, keyed off SM (compute capability) major version.

    Used to pick default precision (see `runtime.precision`) and to key the engine cache —
    an engine built for one architecture is not portable to another.
    """

    PASCAL = "pascal"  # sm_60/61
    VOLTA = "volta"  # sm_70
    TURING = "turing"  # sm_75
    AMPERE = "ampere"  # sm_80/86
    ADA = "ada"  # sm_89
    HOPPER = "hopper"  # sm_90
    BLACKWELL = "blackwell"  # sm_100/120
    UNKNOWN = "unknown"


_SM_MAJOR_TO_ARCH: dict[int, GPUArchitecture] = {
    6: GPUArchitecture.PASCAL,
    7: GPUArchitecture.VOLTA,  # sm_75 (Turing) is disambiguated by minor below
    8: GPUArchitecture.AMPERE,  # sm_89 (Ada) is disambiguated by minor below
    9: GPUArchitecture.HOPPER,
    10: GPUArchitecture.BLACKWELL,
    12: GPUArchitecture.BLACKWELL,
}


def _classify(major: int, minor: int) -> GPUArchitecture:
    if major == 7 and minor >= 5:
        return GPUArchitecture.TURING
    if major == 8 and minor == 9:
        return GPUArchitecture.ADA
    return _SM_MAJOR_TO_ARCH.get(major, GPUArchitecture.UNKNOWN)


@dataclass(frozen=True)
class GPUInfo:
    """Snapshot of one detected CUDA device, used for precision selection and cache keys."""

    index: int
    name: str
    architecture: GPUArchitecture
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    cuda_version: str | None
    driver_version: str | None


class NoCUDADeviceError(RuntimeError):
    """Raised when GPU info is requested but no CUDA-capable device/driver is visible."""


def is_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def detect_gpus() -> list[GPUInfo]:
    """Enumerate visible CUDA devices. Returns an empty list if none are available.

    Does not raise on a CPU-only machine — callers that require a GPU should use
    `require_gpu()` instead, which raises `NoCUDADeviceError` with an actionable message.
    """
    if not is_cuda_available():
        return []

    import torch

    infos = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        infos.append(
            GPUInfo(
                index=index,
                name=props.name,
                architecture=_classify(props.major, props.minor),
                compute_capability=(props.major, props.minor),
                total_memory_bytes=props.total_memory,
                cuda_version=torch.version.cuda,
                driver_version=_driver_version(),
            )
        )
    return infos


def _driver_version() -> str | None:
    try:
        import torch

        version = torch._C._cuda_getDriverVersion()
    except Exception:
        return None
    return str(version) if version else None


def require_gpu(index: int = 0) -> GPUInfo:
    """Return info for the requested device, raising a clear error if unavailable.

    This is the entry point `RuntimeManager` uses before any engine build/load — every other
    function in this module is safe to call without a GPU present.
    """
    gpus = detect_gpus()
    if not gpus:
        raise NoCUDADeviceError(
            "No CUDA-capable GPU detected. TensorRT-Wan requires an NVIDIA GPU with a "
            "CUDA-enabled PyTorch build; CPU-only fallback for the DiT engine is not supported."
        )
    for gpu in gpus:
        if gpu.index == index:
            return gpu
    raise NoCUDADeviceError(f"Requested GPU index {index}, but only {len(gpus)} device(s) detected.")
