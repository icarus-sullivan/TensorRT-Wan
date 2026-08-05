from pathlib import Path

from tensorrt_wan.runtime.cache import CacheKey, EngineCache


def _key(**overrides) -> CacheKey:
    defaults = dict(
        model_hash="abc123",
        tensorrt_version="10.0",
        cuda_version="12.4",
        gpu_architecture="ada",
        optimization_profile="480x832",
        precision="fp16",
    )
    defaults.update(overrides)
    return CacheKey(**defaults)


def test_cache_miss_when_empty(tmp_path: Path):
    cache = EngineCache(tmp_path)
    assert cache.get(_key()) is None


def test_put_then_get_round_trip(tmp_path: Path):
    cache = EngineCache(tmp_path)
    key = _key()
    path = cache.put(key, b"fake-engine-bytes")
    assert path.exists()
    assert cache.get(key) == path
    assert path.read_bytes() == b"fake-engine-bytes"


def test_different_precision_is_a_different_cache_entry(tmp_path: Path):
    cache = EngineCache(tmp_path)
    cache.put(_key(precision="fp16"), b"fp16-engine")
    assert cache.get(_key(precision="fp8")) is None


def test_clear_removes_all_entries(tmp_path: Path):
    cache = EngineCache(tmp_path)
    cache.put(_key(model_hash="a"), b"engine-a")
    cache.put(_key(model_hash="b"), b"engine-b")
    assert cache.clear() == 2
    assert cache.list() == []


def test_disabled_cache_always_misses(tmp_path: Path):
    cache = EngineCache(tmp_path, enabled=False)
    key = _key()
    cache.put(key, b"engine-bytes")  # put() still writes...
    assert cache.get(key) is None  # ...but get() reports a miss when disabled


def test_digest_is_stable_for_equal_keys():
    assert _key().digest() == _key().digest()
    assert _key().digest() != _key(precision="fp8").digest()
