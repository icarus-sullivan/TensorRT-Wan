"""On-disk TensorRT engine cache with automatic invalidation.

An engine is only valid for the exact (component, model, TensorRT version, CUDA version, GPU
architecture, optimization profile, precision) tuple it was built for. Loading a stale engine
silently would be a correctness bug (wrong precision) or a crash (arch mismatch), so `EngineCache`
keys strictly on all seven and never returns a partial/best-effort match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CacheKey:
    # Confirmed necessary against a real collision on RunPod hardware: `vae_encoder` and
    # `vae_decoder` share one checkpoint file (same `model_hash`) and can easily share the same
    # profile/precision too — without `component`, a decoder build silently got served the
    # encoder's cached engine instead of building its own. See docs/wan2.2_i2v_14b_notes.md.
    component: str
    model_hash: str
    tensorrt_version: str
    cuda_version: str
    gpu_architecture: str
    optimization_profile: str
    precision: str
    # Confirmed necessary against a second, real collision: `optimization_profile` is a *name*
    # string (e.g. "480x832"), not the exporter's actual traced shape — two builds of the same
    # component/profile-name but different exporter-kwargs (e.g. VAEEncoderExporter's `frames=1`
    # vs `frames=81`) hash identically and silently overwrite each other's cache entry, even
    # though the underlying ONNX graphs (and what shapes the resulting engine actually accepts)
    # are completely different. Hit for real on 2026-08-06: a `frames=1` test build clobbered a
    # previous `frames=9` engine under the same digest. See docs/wan2.2_i2v_14b_notes.md.
    input_shape_digest: str = ""

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


class EngineCache:
    """Maps a `CacheKey` to an engine file path under `directory`.

    This only manages *paths and metadata* — it does not build or load engines. That's
    `runtime.manager.RuntimeManager`'s job, which calls `.get()`/`.put()` around its own
    TensorRT build/deserialize calls.
    """

    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled

    def get(self, key: CacheKey) -> Path | None:
        """Return the cached engine path if present and its metadata matches `key` exactly."""
        if not self.enabled:
            return None
        engine_path, meta_path = self._paths(key)
        if not engine_path.exists() or not meta_path.exists():
            return None
        stored = json.loads(meta_path.read_text())
        if stored != asdict(key):
            logger.info("Cache entry %s exists but metadata mismatch; treating as a miss", key.digest())
            return None
        logger.debug("Engine cache hit: %s", engine_path)
        return engine_path

    def put(self, key: CacheKey, engine_bytes: bytes) -> Path:
        """Write engine bytes + metadata for `key`, creating the cache directory if needed."""
        self.directory.mkdir(parents=True, exist_ok=True)
        engine_path, meta_path = self._paths(key)
        engine_path.write_bytes(engine_bytes)
        meta_path.write_text(json.dumps(asdict(key), indent=2))
        logger.info("Cached engine at %s", engine_path)
        return engine_path

    def invalidate(self, key: CacheKey) -> None:
        engine_path, meta_path = self._paths(key)
        engine_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    def clear(self) -> int:
        """Delete every cached engine. Returns the number of entries removed."""
        if not self.directory.exists():
            return 0
        removed = 0
        for engine_path in self.directory.glob("*.engine"):
            engine_path.unlink(missing_ok=True)
            engine_path.with_suffix(".json").unlink(missing_ok=True)
            removed += 1
        return removed

    def list(self) -> list[dict[str, str]]:
        if not self.directory.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(self.directory.glob("*.json"))]

    def _paths(self, key: CacheKey) -> tuple[Path, Path]:
        digest = key.digest()
        return self.directory / f"{digest}.engine", self.directory / f"{digest}.json"
