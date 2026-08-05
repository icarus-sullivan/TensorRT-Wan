# TensorRT-Wan

TensorRT-Wan is an open-source acceleration framework for [Wan](https://github.com/Wan-Video) video
generation models. It optimizes the shared Wan DiT backbone with TensorRT — once, not per-workflow —
so every workflow built on Wan (text-to-video, image-to-video, video-to-video, ControlNet, IP-Adapter,
LoRA, and future conditioning methods) benefits from the same accelerated runtime.

> **Status:** pre-alpha, structure/scaffolding phase. See [PLAN.md](PLAN.md) for the full spec.
> No engines are built and no inference runs yet — see [Development Status](#development-status).

## Why one engine, not many

Most TensorRT integrations for diffusion models build a separate engine per workflow. TensorRT-Wan
instead optimizes the single Wan DiT backbone and routes every workflow's conditioning through one
[`ConditioningManager`](tensorrt_wan/conditioning) into one [`DiTEngine`](tensorrt_wan/engine/dit_engine.py).
Adding a new conditioning source (a new ControlNet variant, a new adapter) means adding a
`ConditioningSource`, not a new TensorRT engine.

```
Prompt → Text Encoder ─┐
Image ──────────────────┼─▶ Conditioning Manager ─▶ Unified TensorRT DiT Engine ─▶ TensorRT VAE Decoder ─▶ Video
Control/IP-Adapter/LoRA ┘
```

See [docs/architecture.md](docs/architecture.md) for the full design.

## Installation

```bash
pip install -e ".[dev]"          # core + dev tooling
pip install -e ".[tensorrt]"     # + TensorRT / onnxruntime-gpu (requires an NVIDIA GPU + CUDA)
```

See [docs/installation.md](docs/installation.md) for GPU/driver/TensorRT version requirements.

## Quickstart (standalone Python API)

```python
from tensorrt_wan import WanEngine

engine = WanEngine.from_pretrained("Wan2.1-T2V-14B", precision="auto")
video = engine.generate(prompt="a fox running through snow", num_frames=81, resolution=(480, 832))
video.save("out.mp4")
```

See [docs/python_api.md](docs/python_api.md).

## ComfyUI

Copy or symlink `comfyui/` into `ComfyUI/custom_nodes/tensorrt_wan_comfyui` (avoid naming the
folder `tensorrt_wan` — that would collide with the pip-installed `tensorrt_wan` package these
nodes import). See [docs/comfyui_integration.md](docs/comfyui_integration.md) for node reference
and workflow migration.

## CLI

```bash
trtwan gpu-report                              # detect GPU + supported precisions
trtwan export onnx --model wan2.1-t2v-14b      # PyTorch -> ONNX
trtwan build engine --onnx dit.onnx --profile 480x832   # ONNX -> TensorRT
trtwan cache list / trtwan cache clear
```

See [docs/export.md](docs/export.md) and [docs/engine_generation.md](docs/engine_generation.md).

## Development status

This repository is being built in a structure-first phase: interfaces, module boundaries, exporters,
plugin scaffolding, CLI, ComfyUI nodes, and tests exist, but **no model has been exported, no TensorRT
engine has been built, and no inference has been executed or profiled in this environment.** That
validation happens on GPU-equipped hardware (see [docs/roadmap.md](docs/roadmap.md)). Expect
`NotImplementedError` at the PyTorch→ONNX→TensorRT boundary until that phase.

## Repository layout

| Path | Purpose |
|---|---|
| `tensorrt_wan/runtime` | GPU/TensorRT capability detection, precision selection, engine cache, fallback |
| `tensorrt_wan/conditioning` | Unified conditioning manager (text/image/control/adapter/LoRA/future) |
| `tensorrt_wan/scheduler` | GPU-resident diffusion scheduler state |
| `tensorrt_wan/engine` | Text encoder, unified DiT, VAE encoder/decoder TensorRT engine wrappers |
| `tensorrt_wan/export` | torch.export → ONNX → TensorRT exporters |
| `tensorrt_wan/plugins` | TensorRT plugin registry + CUDA/C++ plugin sources |
| `tensorrt_wan/config` | JSON/YAML configuration schema + loader |
| `tensorrt_wan/api` | Standalone `WanEngine` Python API |
| `tensorrt_wan/cli` | `trtwan` command-line tool |
| `comfyui/` | ComfyUI custom node package |
| `tests/` | Unit/structure tests (no GPU required to collect) |
| `examples/` | Example scripts (Python API + ComfyUI workflow JSON) |
| `docs/` | Architecture, installation, and developer documentation |
| `scripts/` | Plugin build scripts, env setup (not executed by this repo automatically) |

## License

Apache 2.0 — see [LICENSE](LICENSE). Contributions welcome, see [docs/contributing.md](docs/contributing.md).
