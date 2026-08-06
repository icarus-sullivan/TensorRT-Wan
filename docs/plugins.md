# TensorRT Plugins

Custom TensorRT plugins for Wan DiT operations that ONNX/TensorRT don't natively cover. Source
lives in [`tensorrt_wan/plugins/csrc/`](../tensorrt_wan/plugins/csrc/); the Python side
([`plugins/registry.py`](../tensorrt_wan/plugins/registry.py)) just loads the compiled shared
library and validates plugin names.

## Plugins

| Plugin | `kName` | Purpose |
|---|---|---|
| [`rotary_embedding/`](../tensorrt_wan/plugins/csrc/rotary_embedding/) | `RotaryEmbedding` | RoPE applied to attention Q/K |
| [`adalayernorm/`](../tensorrt_wan/plugins/csrc/adalayernorm/) | `AdaLayerNorm` | AdaLN-Zero modulation (DiT-style conditioning) |
| [`custom_attention/`](../tensorrt_wan/plugins/csrc/custom_attention/) | `CustomAttention` | Dispatcher over FlashAttention/FlashAttention-2/3/SageAttention |
| [`patch_embed/`](../tensorrt_wan/plugins/csrc/patch_embed/) | `PatchEmbed` | Latent video -> DiT input tokens |
| [`patch_reconstruct/`](../tensorrt_wan/plugins/csrc/patch_reconstruct/) | `PatchReconstruct` | DiT output tokens -> latent video (inverse of PatchEmbed) |
| [`time_embedding/`](../tensorrt_wan/plugins/csrc/time_embedding/) | `TimeEmbedding` | Sinusoidal timestep embedding |
| [`video_ops/`](../tensorrt_wan/plugins/csrc/video_ops/) | `TemporalResize` | Nearest-neighbor temporal upsampling (VAE decoder) |
| [`activation/`](../tensorrt_wan/plugins/csrc/activation/) | `FusedActivation` | SiLU/GELU/QuickGELU |

## Shared boilerplate

Every plugin extends [`common/plugin_base.h`](../tensorrt_wan/plugins/csrc/common/plugin_base.h)
(`IPluginV2DynamicExt` boilerplate: format support, workspace sizing, serialize/clone/destroy) and
registers via [`common/plugin_creator_base.h`](../tensorrt_wan/plugins/csrc/common/plugin_creator_base.h)
(`IPluginCreator` boilerplate, templated over the plugin type). A plugin implementation only needs
to provide `getOutputDimensions()`, `enqueue()`, and its own (de)serialization — see
`rotary_embedding/plugin.h` for the fullest-documented example of the pattern.

## Building

```bash
TENSORRT_ROOT=/path/to/TensorRT bash scripts/build_plugins.sh
```

Produces `tensorrt_wan/plugins/csrc/build/libtensorrt_wan_plugins.so`. Requires a CUDA toolkit and
the TensorRT SDK — not built as part of `pip install`, since this development phase doesn't
assume GPU hardware is present (PLAN.md's development rule).

## Numerical validation status

The kernels in this repository (AdaLN-Zero, sinusoidal timestep embedding, patch
embed/reconstruct as tiled linear projections, nearest-neighbor temporal resize, standard
activation formulas) implement the generic, well-established versions of each operation as used
across the DiT model family. **None have been validated against Wan's specific reference
implementation** — this repository was built without access to Wan's own source (see
[roadmap.md](roadmap.md)'s validation phase). Before trusting a built engine's output:

1. Build each plugin's op in isolation and compare against the equivalent PyTorch op on the same
   input, per RoPE/AdaLN/etc.
2. **`rotary_embedding` was confirmed wrong and has been rewritten, but still needs numeric
   validation:** it previously implemented rotate-half, while ComfyUI's real Wan implementation
   (`comfy/ldm/flux/math.py`'s `_apply_rope1`, called directly by `comfy/ldm/wan/model.py`) uses
   interleaved-pair rotation — confirmed by reading the actual source on RunPod hardware, see
   [wan2.2_i2v_14b_notes.md](wan2.2_i2v_14b_notes.md#onnx-export-rmsnorm-and-rope-both-needed-fixes).
   `kernel.cu` now rotates adjacent pairs (x[2i], x[2i+1]) per `_apply_rope1`'s math, matching
   `examples/loaders/wan_comfyui_loader.py`'s cloned reference — still needs a build + isolated
   numeric comparison against that reference on GPU hardware before it's trusted in an engine.
   Also check `patch_embed`/`patch_reconstruct` (patch ordering conventions vary too, not yet
   confirmed either way).
3. `custom_attention` has no backend wired up at all yet (see
   [`custom_attention/kernel_dispatch.cpp`](../tensorrt_wan/plugins/csrc/custom_attention/kernel_dispatch.cpp))
   — it raises rather than silently computing something wrong.

## Adding a new plugin

1. Add a `<op_name>/` directory under `plugins/csrc/` with `plugin.h`/`plugin.cpp` (and a
   `kernel.cu` if it needs one), following `rotary_embedding/` as a template.
2. Add the plugin's `.h`/`.cpp`/`.cu` files to `plugins/csrc/CMakeLists.txt`.
3. Add its `kName` string to `PLUGIN_NAMES` in
   [`plugins/registry.py`](../tensorrt_wan/plugins/registry.py).
