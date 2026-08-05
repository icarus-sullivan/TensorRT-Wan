# Engine Generation (ONNX -> TensorRT)

## Building an engine

```bash
trtwan build engine \
  --component dit \
  --onnx dit.onnx \
  --loader my_wan_adapter:load_dit \
  --checkpoint /path/to/wan2.1-t2v-14b \
  --exporter-kwargs '{"in_channels": 36, "text_dim": 4096}' \
  --resolutions 480x832,720x1280 \
  --precision auto
```

This calls [`export/trt_build.py`](../tensorrt_wan/export/trt_build.py)'s
`build_tensorrt_engine()`: one `IOptimizationProfile` per `--resolutions` entry, so a single
engine serves every listed shape (PLAN.md's "static engines for common resolutions" +
"dynamic shapes where practical", reconciled by building one engine with several fixed profiles
rather than either one engine per shape or one fully-dynamic engine).

The `--loader`/`--checkpoint`/`--exporter-kwargs` flags reconstruct the same `ModelExporter`
instance used for the ONNX export (see [export.md](export.md)) purely for its shape/dynamic-axis
metadata — the ONNX file itself is what actually gets parsed and compiled.

## Caching

Successful builds are stored via [`EngineCache`](../tensorrt_wan/runtime/cache.py), keyed on:

- model hash (checkpoint identity)
- TensorRT version
- CUDA version
- GPU architecture (see `runtime.gpu.GPUArchitecture`)
- optimization profile set
- precision

Any of those changing invalidates the cache entry — an engine built for Ada FP16 is never handed
back for an Ampere or FP8 request. Rebuild with `--force`, or clear everything with
`trtwan cache clear`.

## Precision

`--precision auto` defers to [`runtime/precision.py`](../tensorrt_wan/runtime/precision.py)'s
per-architecture default (see [optimization_strategy.md](optimization_strategy.md)). Pin a value
(`fp16`, `fp8`, `bf16`, `fp32`) to override.

## Inspecting a built engine

```bash
trtwan inspect <path-or-cache-digest-prefix>
trtwan list engines
trtwan optimization-report
```

`inspect` prints the cache metadata and, if the `tensorrt` package is importable, every I/O
tensor's shape/dtype (see [`cli/commands/inspect.py`](../tensorrt_wan/cli/commands/inspect.py)).

## What this repository does NOT do (yet)

No engine has been built in this repository — see PLAN.md's development rule and
[roadmap.md](roadmap.md) for when that happens (RunPod GPU validation phase).
