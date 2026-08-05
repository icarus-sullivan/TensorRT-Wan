from pathlib import Path

from tensorrt_wan.config.loader import default_config, load_config, save_config
from tensorrt_wan.config.schema import PrecisionConfig, ResolutionProfile, TensorRTWanConfig


def test_default_config_has_default_resolution_profiles():
    config = default_config()
    names = {p.name for p in config.resolution_profiles}
    assert "480x832" in names
    assert "1080x1920" in names


def test_json_round_trip(tmp_path: Path):
    config = TensorRTWanConfig(precision=PrecisionConfig(mode="fp8", allow_fp8=False))
    path = tmp_path / "config.json"
    save_config(config, path)

    loaded = load_config(path)
    assert loaded.precision.mode == "fp8"
    assert loaded.precision.allow_fp8 is False


def test_yaml_round_trip(tmp_path: Path):
    config = TensorRTWanConfig(
        resolution_profiles=[ResolutionProfile("custom", 100, 200, num_frames=17)]
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)

    loaded = load_config(path)
    assert len(loaded.resolution_profiles) == 1
    assert loaded.resolution_profiles[0] == ResolutionProfile("custom", 100, 200, num_frames=17)


def test_load_config_ignores_unknown_keys(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text('{"precision": {"mode": "fp16"}, "some_future_field": 123}')
    loaded = load_config(path)
    assert loaded.precision.mode == "fp16"


def test_load_config_rejects_unknown_extension(tmp_path: Path):
    path = tmp_path / "config.txt"
    path.write_text("{}")
    try:
        load_config(path)
        assert False, "expected ValueError"
    except ValueError:
        pass
