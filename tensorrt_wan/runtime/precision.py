"""Precision selection.

Implements the project's precision strategy: pick the highest-performance precision that does
not measurably degrade output quality, never drop precision purely to save memory, and default
per-architecture when the user hasn't pinned a mode.

    Blackwell -> FP8 where a per-op quality check clears it, FP16 otherwise
    Hopper/Ada/Ampere/Turing and earlier -> FP16
"""

from __future__ import annotations

from dataclasses import dataclass

from tensorrt_wan.config.schema import PrecisionConfig
from tensorrt_wan.runtime.gpu import GPUArchitecture, GPUInfo

_ARCHITECTURE_MAX_PRECISION: dict[GPUArchitecture, str] = {
    GPUArchitecture.BLACKWELL: "fp8",
    GPUArchitecture.HOPPER: "fp16",
    GPUArchitecture.ADA: "fp16",
    GPUArchitecture.AMPERE: "fp16",
    GPUArchitecture.TURING: "fp16",
    GPUArchitecture.VOLTA: "fp16",
    GPUArchitecture.PASCAL: "fp32",
    GPUArchitecture.UNKNOWN: "fp16",
}


@dataclass(frozen=True)
class PrecisionDecision:
    """Result of `select_precision`, with the reasoning kept alongside the answer.

    `reason` is surfaced in logs/diagnostics — precision selection silently picking the wrong
    thing is exactly the kind of bug that's invisible without an audit trail.
    """

    precision: str
    reason: str


def select_precision(gpu: GPUInfo, config: PrecisionConfig) -> PrecisionDecision:
    if config.mode != "auto":
        return PrecisionDecision(
            precision=config.mode,
            reason=f"explicit precision.mode={config.mode!r} in config",
        )

    ceiling = _ARCHITECTURE_MAX_PRECISION.get(gpu.architecture, "fp16")

    if ceiling == "fp8" and not config.allow_fp8:
        return PrecisionDecision(
            precision="fp16",
            reason=f"{gpu.architecture.value} supports fp8 but precision.allow_fp8=False",
        )
    return PrecisionDecision(
        precision=ceiling,
        reason=f"auto: {gpu.architecture.value} (sm_{gpu.compute_capability[0]}{gpu.compute_capability[1]}) default ceiling",
    )
