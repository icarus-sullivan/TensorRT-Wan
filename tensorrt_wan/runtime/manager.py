"""RuntimeManager: the single object that owns GPU/TensorRT state for a process.

Every engine wrapper (`engine.dit_engine`, `engine.vae_engine`, `engine.text_encoder_engine`)
takes a `RuntimeManager` rather than probing the GPU/TensorRT/cache itself — this is what keeps
"only one place does GPU detection" true as the module count grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tensorrt_wan.config.schema import TensorRTWanConfig
from tensorrt_wan.runtime.cache import EngineCache
from tensorrt_wan.runtime.capability import TensorRTCapability, detect_tensorrt
from tensorrt_wan.runtime.gpu import GPUInfo, detect_gpus, is_rocm_available
from tensorrt_wan.runtime.precision import PrecisionDecision, select_precision
from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_backend(gpus: list[GPUInfo], tensorrt: TensorRTCapability) -> str:
    """Which engine backend this process should build/load against: `"migraphx"` when TensorRT
    isn't available and at least one detected GPU is a ROCm/AMD device, `"tensorrt"` otherwise.

    TensorRT has no ROCm build at all (see docs/rocm_setup.md) -- on AMD hardware there is
    nothing else to fall back to, so this doesn't try to be cleverer than a two-way switch.
    Neither-available (no GPU, or an NVIDIA GPU with no TensorRT installed) still resolves to
    `"tensorrt"` — existing NVIDIA-only error paths (e.g. `gpu.require_gpu()`) apply unchanged.
    """
    if not tensorrt.available and is_rocm_available():
        return "migraphx"
    return "tensorrt"


@dataclass
class DiagnosticsReport:
    gpus: list[GPUInfo]
    tensorrt: TensorRTCapability
    backend: str
    precision: PrecisionDecision | None
    loaded_plugins: list[str]
    cache_entries: int

    def as_text(self) -> str:
        lines = [
            f"GPUs detected: {len(self.gpus)}",
            *[
                f"  [{g.index}] {g.name} ({g.architecture.value}, sm_{g.compute_capability[0]}{g.compute_capability[1]}, "
                f"{g.total_memory_bytes / (1 << 30):.1f} GiB)"
                for g in self.gpus
            ],
            f"TensorRT: {'available ' + self.tensorrt.version if self.tensorrt.available else 'not installed'}",
        ]
        if self.tensorrt.available:
            lines.append(
                f"  FP8={self.tensorrt.supports_fp8} BF16={self.tensorrt.supports_bf16} "
                f"strongly_typed={self.tensorrt.supports_strongly_typed}"
            )
        lines.append(f"Backend: {self.backend}")
        if self.precision is not None:
            lines.append(f"Selected precision: {self.precision.precision} ({self.precision.reason})")
        lines.append(f"Loaded plugins: {', '.join(self.loaded_plugins) or '(none)'}")
        lines.append(f"Cached engines: {self.cache_entries}")
        return "\n".join(lines)


class RuntimeManager:
    """Detects hardware/software capability once and hands out decisions/resources from it.

    Construction is cheap and side-effect-free beyond detection (no engines are loaded, no
    workspace is allocated) so it's safe to instantiate for a `gpu-report`/diagnostics CLI call
    even with no GPU present.
    """

    def __init__(self, config: TensorRTWanConfig | None = None) -> None:
        self.config = config or TensorRTWanConfig()
        self.gpus: list[GPUInfo] = detect_gpus()
        self.tensorrt: TensorRTCapability = detect_tensorrt()
        self.backend: str = _resolve_backend(self.gpus, self.tensorrt)
        self.cache = EngineCache(self.config.cache.directory, enabled=self.config.cache.enabled)
        self._loaded_plugins: list[str] = []
        self._loaded_engines: dict[str, object] = {}

    @property
    def primary_gpu(self) -> GPUInfo | None:
        return self.gpus[0] if self.gpus else None

    def select_precision(self, gpu_index: int = 0) -> PrecisionDecision:
        if not self.gpus:
            raise RuntimeError("select_precision() requires a detected GPU; none found.")
        gpu = next((g for g in self.gpus if g.index == gpu_index), self.gpus[0])
        decision = select_precision(gpu, self.config.precision)
        logger.info("Precision selected: %s (%s)", decision.precision, decision.reason)
        return decision

    def load_plugins(self, plugin_names: list[str]) -> None:
        """Register the given TensorRT plugin names via `plugins.registry`.

        Deferred import avoids requiring the `tensorrt` package for any RuntimeManager use
        that doesn't touch plugins (e.g. a CPU-only `gpu-report`).
        """
        from tensorrt_wan.plugins.registry import load_plugin

        for name in plugin_names:
            if self.config.plugins.enabled.get(name, True) is False:
                logger.info("Plugin %s disabled by config; skipping", name)
                continue
            load_plugin(name)
            self._loaded_plugins.append(name)

    def register_engine(self, name: str, engine: object) -> None:
        """Track a loaded engine handle so it can be unloaded/reported on later."""
        self._loaded_engines[name] = engine

    def unload_engine(self, name: str) -> None:
        self._loaded_engines.pop(name, None)

    def unload_all(self) -> None:
        self._loaded_engines.clear()

    def diagnostics(self) -> DiagnosticsReport:
        precision = self.select_precision() if self.gpus else None
        return DiagnosticsReport(
            gpus=self.gpus,
            tensorrt=self.tensorrt,
            backend=self.backend,
            precision=precision,
            loaded_plugins=list(self._loaded_plugins),
            cache_entries=len(self.cache.list()),
        )
