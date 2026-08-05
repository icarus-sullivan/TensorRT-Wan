# ComfyUI Integration

## Install

Copy or symlink `comfyui/` into `ComfyUI/custom_nodes/tensorrt_wan_comfyui`:

```bash
ln -s /path/to/TensorRT-Wan/comfyui ComfyUI/custom_nodes/tensorrt_wan_comfyui
```

**Do not name the folder `tensorrt_wan`.** ComfyUI imports each `custom_nodes/<folder>` as a
top-level module named after the folder; naming it `tensorrt_wan` would collide with the
pip-installed `tensorrt_wan` package these nodes import, and one would shadow the other depending
on import order. Every module in `comfyui/` uses relative imports internally for the same reason
— the package works regardless of what name ComfyUI assigns it, but the pip package name is not
negotiable. See [`comfyui/__init__.py`](../comfyui/__init__.py).

The pip package must be installed (`pip install -e .` from this repo) into the same Python
environment ComfyUI runs in.

## Nodes

| Node | Class | Inputs -> Outputs |
|---|---|---|
| TensorRT Runtime Manager | `TensorRTRuntimeManager` | precision, cache_dir -> runtime |
| TensorRT Precision Selector | `TensorRTPrecisionSelector` | runtime, gpu_index -> runtime, precision (STRING) |
| TensorRT Wan Loader | `TensorRTWanLoader` | runtime, model_dir -> model_config, text_encoder, dit, vae_encoder, vae_decoder |
| TensorRT Engine Builder | `TensorRTEngineBuilder` | runtime, component, loader, checkpoint_path, ... -> runtime, engine_path |
| TensorRT Text Encoder | `TensorRTTextEncoder` | text_encoder, prompt -> conditioning |
| TensorRT VAE Encoder | `TensorRTVAEEncoder` | vae_encoder, image, role -> conditioning |
| TensorRT Conditioning Manager | `TensorRTConditioningManager` | text/image/control/ip_adapter/lora (all optional) -> conditioning |
| TensorRT Scheduler | `TensorRTScheduler` | shift -> scheduler |
| TensorRT Sampler | `TensorRTSampler` | dit, conditioning, scheduler, latent, steps, cfg, seed -> latent |
| TensorRT VAE Decoder | `TensorRTVAEDecoder` | vae_decoder, latent -> image |
| TensorRT Cache Manager | `TensorRTCacheManager` | runtime, action -> runtime, report (STRING) |
| TensorRT Diagnostics | `TensorRTDiagnostics` | runtime -> runtime, report (STRING) |
| TensorRT Engine Inspector | `TensorRTEngineInspector` | engine_path -> report (STRING) |

Source: [`comfyui/nodes/`](../comfyui/nodes/).

## Socket types

Nodes with a direct ComfyUI equivalent reuse ComfyUI's own types (`LATENT`, `IMAGE`) so
TensorRT-Wan nodes connect directly to stock nodes — a `TensorRTVAEDecoder` output plugs straight
into a stock `SaveImage`, and a `TensorRTSampler`'s `LATENT` input can come from a stock
"Empty Latent Video"-equivalent node. TensorRT-Wan-specific handles (a loaded runtime, an engine,
merged conditioning) get their own `TRTWAN_*` socket types (see
[`comfyui/types.py`](../comfyui/types.py)) so ComfyUI's graph validation catches a mismatched
connection at graph-build time instead of at run time.

## A typical T2V graph

```
TensorRT Runtime Manager ---> TensorRT Wan Loader ---> TensorRT Text Encoder ---> TensorRT Conditioning Manager
                                       |                                                    |
                                       +-> TensorRT Scheduler --------+                     |
                                       |                              v                     v
                                       +-> TensorRT Sampler <---------+---------------------+
                                       |         |
                                       +-> TensorRT VAE Decoder <-----+
                                                  |
                                                  v
                                            (stock SaveImage / VHS Video Combine)
```

For I2V, additionally wire an image through `TensorRT VAE Encoder` (role=`first_frame`) into the
Conditioning Manager's `image` socket.

## Migrating an existing Wan workflow

Per PLAN.md's compatibility goal, only four node types typically need replacing:

1. Model loader -> `TensorRT Wan Loader`
2. Sampler -> `TensorRT Sampler`
3. VAE (encode/decode) -> `TensorRT VAE Encoder` / `TensorRT VAE Decoder`
4. Scheduler -> `TensorRT Scheduler`

Everything else in the workflow (prompt nodes, image loaders, `SaveImage`/video-combine nodes)
stays as-is.

## Status

No engines are built by this repository yet (see [roadmap.md](roadmap.md)), so
`TensorRT Wan Loader` will raise `FileNotFoundError` until you've run
`trtwan build engine`/`TensorRT Engine Builder` against a real checkpoint on GPU hardware.
