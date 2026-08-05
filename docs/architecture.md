# Architecture

## Principle: one engine, not one per workflow

TensorRT-Wan optimizes the Wan DiT backbone once. Every workflow (T2V, I2V, V2V, ControlNet,
IP-Adapter, LoRA, future conditioning methods) routes through the same
[`DiTEngine`](../tensorrt_wan/engine/dit_engine.py) — workflow differences live entirely in what
gets fed into it, not in a separate engine per workflow.

```
                    Prompt
                       |
                Text Encoder                (engine/text_encoder_engine.py)
                       |
                       v
             Conditioning Manager           (conditioning/manager.py)
          +--------+---------+----------+
          |        |         |          |
       Text     Image     Control     Future        (conditioning/sources/*.py)
                  |         |             |
                  v         v             v
              Unified Conditioning Manager
                       |
                       v
         Unified TensorRT DiT Engine        (engine/dit_engine.py)
                       |
                       v
                 Latent Output
                       |
                       v
              TensorRT VAE Decoder           (engine/vae_engine.py)
                       |
                       v
                    Video
```

## Module map

| Module | Responsibility | Key files |
|---|---|---|
| `runtime` | GPU/TensorRT detection, precision selection, engine cache, plugin loading, fallback | `runtime/manager.py`, `runtime/gpu.py`, `runtime/precision.py`, `runtime/cache.py`, `runtime/fallback.py` |
| `conditioning` | Fan-in of every conditioning source into one payload | `conditioning/manager.py`, `conditioning/source.py`, `conditioning/sources/*.py` |
| `scheduler` | GPU-resident diffusion sampling loop state | `scheduler/base.py`, `scheduler/flow_match.py`, `scheduler/state.py` |
| `engine` | TensorRT engine wrappers (text encoder, DiT, VAE enc/dec) | `engine/base.py`, `engine/dit_engine.py`, `engine/text_encoder_engine.py`, `engine/vae_engine.py` |
| `export` | PyTorch -> torch.export -> ONNX -> TensorRT pipeline | `export/pipeline.py`, `export/torch_export.py`, `export/onnx_export.py`, `export/trt_build.py`, `export/exporters/*.py` |
| `plugins` | Custom TensorRT ops for operations ONNX/TensorRT don't natively support | `plugins/registry.py`, `plugins/csrc/*` |
| `config` | JSON/YAML-serializable configuration schema | `config/schema.py`, `config/loader.py` |
| `api` | Standalone Python API (`WanEngine`) | `api/wan_engine.py` |
| `cli` | `trtwan` command-line tool | `cli/main.py`, `cli/commands/*.py` |

`comfyui/` (top-level, not under `tensorrt_wan/`) is a thin ComfyUI node wrapper around the same
`RuntimeManager`/`ConditioningManager`/engine classes — see
[comfyui_integration.md](comfyui_integration.md).

## Why conditioning is a registry, not a branch

`ConditioningManager` (see [`conditioning/manager.py`](../tensorrt_wan/conditioning/manager.py))
holds a `dict[ConditioningKind, ConditioningSource]`. Adding IP-Adapter support was adding
`IPAdapterConditioningSource` and registering it — no change to `DiTEngine`, no new TensorRT
engine, no branch in the sampling loop. `DiTEngine.denoise_step` only ever sees the merged
`UnifiedConditioning.embeddings` dict (keyed by `ConditioningKind.value`), so it has no idea how
many conditioning sources contributed to a given call.

LoRA is the one conditioning "source" that doesn't produce an embedding — it produces weight
deltas, filed into `UnifiedConditioning.lora_weights` instead of `.embeddings` (see
`ConditioningManager._merge`). Those deltas are expected to already be folded into the loaded
engine's weights at build time; `denoise_step` accepts them for bookkeeping/diagnostics, not to
apply them per-step.

## Why the scheduler is GPU-resident

`SchedulerState` (see [`scheduler/state.py`](../tensorrt_wan/scheduler/state.py)) keeps
`timesteps`/`sigmas` as device tensors and only the loop-control `step_index` as a plain Python
int. `DiTEngine.generate`'s loop never calls `.item()`/`.cpu()` on a per-step tensor — every
device-to-host sync in a 30+ step sampling loop is latency the GPU scheduler goals in PLAN.md
explicitly rule out.

## Why the export pipeline uses a `ModelExporter` per component

Each of `TextEncoderExporter`/`DiTExporter`/`VAEEncoderExporter`/`VAEDecoderExporter` (see
[`export/exporters/`](../tensorrt_wan/export/exporters/)) only describes example inputs, dynamic
shape ranges, and I/O tensor names for its component. `export/pipeline.py` runs the same
`torch.export -> ONNX -> TensorRT` sequence for all of them. Supporting a new Wan release's
slightly different tensor shapes means editing (or adding) an exporter, not the pipeline.
