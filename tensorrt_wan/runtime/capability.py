"""TensorRT installation/capability detection.

Separate from `runtime.gpu` because a GPU can be present without a working TensorRT install
(wrong CUDA version, package not installed, plugin .so missing) — the two failure modes need
different diagnostics and different fallback behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorRTCapability:
    available: bool
    version: str | None
    supports_fp8: bool
    supports_bf16: bool
    supports_strongly_typed: bool


def detect_tensorrt() -> TensorRTCapability:
    """Probe for a usable `tensorrt` package. Never raises — absence is a valid, common result
    on a dev machine and is handled by `runtime.fallback`, not by an exception here.
    """
    try:
        import tensorrt as trt
    except ImportError:
        return TensorRTCapability(
            available=False,
            version=None,
            supports_fp8=False,
            supports_bf16=False,
            supports_strongly_typed=False,
        )

    version = trt.__version__
    major = int(version.split(".")[0])
    return TensorRTCapability(
        available=True,
        version=version,
        supports_fp8=major >= 9,
        supports_bf16=major >= 9,
        supports_strongly_typed=major >= 9,
    )
