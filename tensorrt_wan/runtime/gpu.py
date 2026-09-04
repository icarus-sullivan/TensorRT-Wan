"""GPU detection.

Detection is deliberately lazy/optional at import time: this module must import cleanly on a
machine with no NVIDIA GPU and no CUDA-enabled torch build (development laptops, CI), and only
raise when a caller actually asks for GPU info that isn't there.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GPUArchitecture(str, Enum):
    """GPU microarchitecture. NVIDIA members are keyed off SM (compute capability) major
    version; AMD members are keyed off ROCm's `gcnArchName` instead, since SM major/minor is
    a CUDA-only concept with no meaningful equivalent on a HIP device.

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
    AMD_RDNA3 = "amd_rdna3"  # gfx11xx (e.g. gfx1103 — Strix/Phoenix-class APUs)
    AMD_UNKNOWN = "amd_unknown"  # any other AMD gfx target


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


def _classify_amd(gcn_arch_name: str) -> GPUArchitecture:
    """Classify an AMD device from ROCm's `gcnArchName` (e.g. `"gfx1103"`), not SM major/minor
    — that's a CUDA-only concept with no meaningful equivalent on a HIP device.

    Unverified against real gfx1103 hardware (no GPU execution happens locally in this project,
    see PLAN.md's dev rule) — matches this file's existing "don't fail, degrade to a safe
    default" philosophy for every untested NVIDIA architecture below Blackwell.
    """
    if gcn_arch_name.startswith("gfx11"):
        return GPUArchitecture.AMD_RDNA3
    return GPUArchitecture.AMD_UNKNOWN


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


def is_rocm_available() -> bool:
    """True on a ROCm-built PyTorch with a visible AMD device.

    ROCm-built PyTorch aliases its HIP backend under the same `torch.cuda.*` namespace
    (`is_available()`/`device_count()`/`get_device_properties()` all work unmodified) — the one
    reliable way to tell it apart from a real CUDA build is `torch.version.hip`, which is `None`
    on a CUDA build and a version string on a ROCm one.
    """
    try:
        import torch

        return bool(torch.cuda.is_available()) and torch.version.hip is not None
    except ImportError:
        return False


def detect_gpus() -> list[GPUInfo]:
    """Enumerate visible CUDA or ROCm devices. Returns an empty list if none are available.

    Does not raise on a CPU-only machine — callers that require a GPU should use
    `require_gpu()` instead, which raises `NoCUDADeviceError` with an actionable message.
    """
    if not is_cuda_available():
        return []

    import torch

    rocm = is_rocm_available()

    infos = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        if rocm:
            architecture = _classify_amd(getattr(props, "gcnArchName", ""))
        else:
            architecture = _classify(props.major, props.minor)
        infos.append(
            GPUInfo(
                index=index,
                name=props.name,
                architecture=architecture,
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
            "No CUDA- or ROCm-capable GPU detected. TensorRT-Wan requires an NVIDIA GPU with a "
            "CUDA-enabled PyTorch build, or an AMD GPU with a ROCm-enabled PyTorch build; "
            "CPU-only fallback for the DiT engine is not supported."
        )
    for gpu in gpus:
        if gpu.index == index:
            return gpu
    raise NoCUDADeviceError(f"Requested GPU index {index}, but only {len(gpus)} device(s) detected.")
