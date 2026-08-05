# Installation

## Requirements

- Python >= 3.10
- An NVIDIA GPU (Ampere or newer recommended; see [supported_gpus.md](supported_gpus.md)) with a
  recent driver, for anything beyond `pip install` and `trtwan gpu-report`
- CUDA toolkit matching your installed `torch`/`tensorrt` build
- TensorRT >= 10.0 for building/running engines (optional for the export-only / structure-only
  parts of the package)

## Core install

```bash
pip install -e .
```

Installs `torch`, `numpy`, `pyyaml`, `onnx` — enough to import `tensorrt_wan`, run
`tensorrt_wan.config`/`tensorrt_wan.runtime.gpu` on a CPU-only machine, and use the CLI's
`gpu-report`/`cache` commands.

## With TensorRT (engine build/run)

```bash
pip install -e ".[tensorrt]"
```

Adds `tensorrt`, `onnxruntime-gpu`, `polygraphy`. Requires a matching CUDA install; see NVIDIA's
own TensorRT installation guide for driver/CUDA compatibility — this project doesn't pin a CUDA
version since it targets multiple GPU generations (PLAN.md's "future NVIDIA GPU architectures"
goal).

## ComfyUI

```bash
pip install -e ".[comfyui]"
```

then copy or symlink `comfyui/` into `ComfyUI/custom_nodes/tensorrt_wan_comfyui` (see
[comfyui_integration.md](comfyui_integration.md) for why the folder must not be named
`tensorrt_wan`). ComfyUI must be able to `import tensorrt_wan` — install into the same Python
environment ComfyUI runs in.

## Dev tooling

```bash
pip install -e ".[dev]"
```

Adds `pytest`, `mypy`, `ruff`, `black`.

## Verifying the install

```bash
trtwan gpu-report
```

Reports detected GPUs, TensorRT availability/version, and the precision that would be selected —
this works without ever building an engine, so it's the right first command to run after install.

## Building the TensorRT plugins

The custom plugins under `tensorrt_wan/plugins/csrc/` are not prebuilt; they require a CUDA
toolkit and the TensorRT SDK headers/libs at build time:

```bash
TENSORRT_ROOT=/path/to/TensorRT bash scripts/build_plugins.sh
```

See [plugins.md](plugins.md).
