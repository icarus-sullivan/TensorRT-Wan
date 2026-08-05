# Python API

## Quickstart

```python
from tensorrt_wan import WanEngine

engine = WanEngine.from_pretrained("/path/to/built/Wan2.1-T2V-14B", precision="auto")
video = engine.generate(prompt="a fox running through snow", num_frames=81, resolution=(480, 832))
video.save("out.mp4")
```

See [`examples/text_to_video.py`](../examples/text_to_video.py) and
[`examples/image_to_video.py`](../examples/image_to_video.py).

## `WanEngine.from_pretrained`

```python
WanEngine.from_pretrained(
    model_dir: str | Path,
    *,
    precision: PrecisionMode = "auto",
    device: torch.device | None = None,
    runtime_config: TensorRTWanConfig | None = None,
    tokenizer: object | None = None,
) -> WanEngine
```

`model_dir` must contain `wan_model.json` (see
[`api/model_config.py`](../tensorrt_wan/api/model_config.py)'s `WanModelConfig`) plus
`text_encoder.engine`, `dit.engine`, `vae_encoder.engine`, `vae_decoder.engine` — the output of
[`trtwan build engine`](engine_generation.md) run once per component. Without a `tokenizer`
argument, a HuggingFace `transformers.AutoTokenizer` is loaded from `wan_model.json`'s
`tokenizer_name` field (requires `transformers` installed).

## `WanEngine.generate`

```python
engine.generate(
    prompt: str,
    *,
    image: torch.Tensor | None = None,       # (1, 3, H, W) in [-1, 1]; set for I2V
    num_frames: int | None = None,            # defaults to the model config's default_num_frames
    resolution: tuple[int, int] | None = None,  # (height, width); defaults from model config
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    seed: int | None = None,
) -> VideoOutput
```

T2V and I2V share this one call — pass `image=` for I2V, omit it for T2V (see
`WanEngine.generate`'s docstring in
[`api/wan_engine.py`](../tensorrt_wan/api/wan_engine.py) for why: both route through the same
`ConditioningManager` and `DiTEngine`, per the project's unified-engine design in
[architecture.md](architecture.md)).

## `VideoOutput`

```python
video.save(path)       # requires `pip install 'imageio[ffmpeg]'`
video.as_numpy()        # (T, H, W, C) uint8
video.frames             # torch.Tensor, same shape, on the generation device
```

See [`api/video_output.py`](../tensorrt_wan/api/video_output.py).

## Building your own pipeline

`WanEngine` composes public building blocks — use them directly for more control (e.g. swapping
the scheduler, adding a conditioning source `WanEngine` doesn't wire up yet):

```python
from tensorrt_wan.runtime import RuntimeManager
from tensorrt_wan.conditioning import ConditioningManager
from tensorrt_wan.conditioning.sources import TextConditioningSource
from tensorrt_wan.scheduler import FlowMatchEulerScheduler
from tensorrt_wan.engine import DiTEngine
```

See [architecture.md](architecture.md) for how these fit together.
