"""Loading a custom precision/cache/resolution configuration.

Equivalent to editing `~/.config/tensorrt_wan/config.yaml` and passing `--config`, but shown here
as plain Python for embedding into a larger application. See docs/architecture.md's
Configuration section for every field.
"""

from pathlib import Path

from tensorrt_wan.config.schema import CacheConfig, PrecisionConfig, ResolutionProfile, TensorRTWanConfig
from tensorrt_wan.runtime.manager import RuntimeManager


def main() -> None:
    config = TensorRTWanConfig(
        precision=PrecisionConfig(mode="fp8", allow_fp8=True),
        cache=CacheConfig(directory=Path("/workspace/trtwan_cache")),
        resolution_profiles=[
            ResolutionProfile("portrait_720p", height=1280, width=720),
            ResolutionProfile("landscape_720p", height=720, width=1280),
        ],
    )

    runtime = RuntimeManager(config)
    print(runtime.diagnostics().as_text())


if __name__ == "__main__":
    main()
