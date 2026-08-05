# Troubleshooting

## `NoCUDADeviceError: No CUDA-capable GPU detected`

Raised by `runtime.gpu.require_gpu()`. Run `trtwan gpu-report` — if it also reports zero devices,
either no NVIDIA GPU is present, the driver isn't installed, or your `torch` build is CPU-only
(`torch.cuda.is_available()` returns `False`). This project doesn't fall back to CPU for the DiT
engine (see `runtime/gpu.py`'s docstring); the text encoder/VAE stages could in principle, but
that's not wired up.

## `FileNotFoundError` from `WanEngine.from_pretrained` / `TensorRTEngineWrapper.load`

Expected until engines are actually built — see [engine_generation.md](engine_generation.md).
Check that `model_dir` contains `wan_model.json` and all four `.engine` files.

## `Plugin library not found at .../csrc/build/libtensorrt_wan_plugins.so`

The plugins aren't prebuilt (see [plugins.md](plugins.md)). Run
`TENSORRT_ROOT=/path/to/TensorRT bash scripts/build_plugins.sh` first.

## A TensorRT op silently falls back to PyTorch and generation is slower than expected

Check the logs at `INFO` or above — `runtime.fallback.run_with_fallback` (used by every engine
call, see [`engine/base.py`](../tensorrt_wan/engine/base.py)) logs a `WARNING` naming the failed
op and the underlying exception every time it falls back. It never crashes silently, but it also
never fails silently — a fallback that's firing every generation is a signal something is
misconfigured (wrong precision for the engine's build, a plugin that isn't loaded via
`RuntimeManager.load_plugins`, a shape outside the built optimization profile).

## Cache seems to be serving a stale engine

It can't — `EngineCache` keys strictly on model hash + TensorRT version + CUDA version + GPU
architecture + optimization profile + precision (see
[engine_generation.md](engine_generation.md#caching)); any mismatch is a cache miss, not a
partial/best-effort hit. If you changed something not in that key (e.g. edited the exporter's
`example_inputs()` without changing dimensions), rebuild with `--force` or `trtwan cache clear`.

## `ImportError` on `transformers` / `imageio` when using the standalone API

Both are optional: `transformers` for the default tokenizer loader in
`WanEngine.from_pretrained` (pass your own `tokenizer=` to avoid it), `imageio[ffmpeg]` for
`VideoOutput.save()` (use `.as_numpy()` / `.frames` directly to avoid it).

## Nothing here explains a real engine-build or inference failure

This repository has not built an engine or run inference in this environment (see PLAN.md's
development rule) — this page covers structural/config errors reachable without a GPU. Numerical
and performance issues discovered on RunPod GPU hardware during the validation phase (see
[roadmap.md](roadmap.md)) will be added here as they're found.
