# Wan 2.2 I2V 14B — checkpoint inspection notes

Findings from directly inspecting the safetensors header (no `tensorrt` / GPU required — see the
inspection commands at the bottom) of the checkpoints staged on the RunPod instance at
`/workspace/runpod-slim/ComfyUI/models/diffusion_models/`:

- `wan2.2_i2v_high_noise_14B_fp16.safetensors`
- `wan2.2_i2v_low_noise_14B_fp16.safetensors` (same architecture, separate expert — see MoE note below)

Recorded here because the models aren't always available to re-inspect (ephemeral RunPod
instance) — this is the ground truth `ModelExporter`/`--loader` work should be built against,
without re-deriving it from scratch each time.

## Architecture (decoded from tensor names/shapes)

| Param | Value | Source |
|---|---|---|
| `num_layers` | 40 | `max(blocks.N.*)` + 1 |
| hidden dim | 5120 | `blocks.0.self_attn.q.weight` shape `[5120, 5120]` |
| ffn dim | 13824 | `blocks.0.ffn.0.weight` shape `[13824, 5120]` |
| patch size | `(1, 2, 2)` (T, H, W) | `patch_embedding.weight` shape `[5120, 36, 1, 2, 2]` |
| DiT **input** channels | **36** | same tensor, dim 1 |
| DiT **output** (latent) channels | **16** | `head.head.weight [64, 5120]`; 64 = 16 × 1×2×2 |
| text embedding input dim | 4096 | `text_embedding.0.weight [5120, 4096]` — matches UMT5-XXL hidden size |
| time embedding input dim | 256 | `time_embedding.0.weight [5120, 256]` |
| AdaLN modulation (per block) | 6-way, `[1, 6, 5120]` | `blocks.0.modulation` |
| AdaLN modulation (final head) | 2-way, `[1, 2, 5120]` | `head.modulation` |
| cross-attn image path (`img_emb.*`) | **absent** | no matching keys anywhere in the checkpoint |

## The important part: how I2V conditioning actually works here

**36 input channels vs. 16 output channels means Wan I2V conditions by channel-concatenation,
not cross-attention.** The DiT's input is `noise_latent (16ch) ++ image_latent (16ch) ++ mask
(4ch) = 36ch`, concatenated *before* `patch_embedding`. There is no `img_emb`-style projection
feeding a separate cross-attention path (that pattern exists in some other video model families,
e.g. IP-Adapter-style conditioning, but not in this checkpoint).

**This does not match what's currently built in this repo.** `export/exporters/dit.py`'s
`DiTExporter.example_inputs()` only has a 16-channel `latents` input plus a separate `text`
tensor — no 36-channel concatenated input, no image-latent/mask inputs at all.
`conditioning/sources/image.py`'s `ImageConditioningSource` (kind=`FIRST_FRAME`) produces a
`ConditioningTensor.embedding`, which `ConditioningManager` currently files into
`UnifiedConditioning.embeddings["first_frame"]` — i.e. treated like a separate cross-attention
embedding, which is architecturally wrong for real Wan I2V.

**Before writing a working I2V exporter, this needs fixing:**
1. `DiTExporter` needs a 36-channel `latents` input (or an explicit `image_latent`/`mask` input
   pair that gets concatenated before the traced forward call).
2. `DiTEngine._build_inputs` (`engine/dit_engine.py`) needs to concatenate
   `conditioning.embeddings["first_frame"]` (+ a mask tensor) onto `latents` on the channel axis,
   instead of passing it as a same-named engine input tensor.
3. `PatchEmbedPlugin` (`plugins/csrc/patch_embed/`) already takes `channels` as a runtime
   parameter, so the plugin itself doesn't need to change — just what gets passed into it.

T2V-only export (no image conditioning) is unaffected by this — a pure 16-channel `latents`
input with no image conditioning still matches the exporter as built today.

## MoE (high/low noise experts)

Confirmed separately (see prior conversation, not re-derived from the checkpoint itself): Wan 2.2
uses two full 14B experts (`high_noise` / `low_noise`, likely each following the exact same
architecture table above — not yet confirmed the low-noise checkpoint's tensor shapes match
byte-for-byte identically, worth a quick header diff before assuming so), switched mid-schedule
by a boundary parameter. `DiTEngine`/`scheduler` in this repo assume **one** engine for the whole
run — no expert-switching logic exists yet. Needs a design decision (two engines + switch logic
in the sampling loop, vs. scoping this repo to Wan 2.1-style single-expert models) before I2V
export is worth pursuing on the 2.2 checkpoints specifically.

## Reusing ComfyUI's own model code (avoids reimplementing the architecture)

ComfyUI already has a working, checkpoint-loading Wan implementation in this environment —
confirmed present, not yet fully wired into a `--loader`:

- `comfy/model_base.py`: `class WAN22(WAN21)` at (approx) line 1835, `class WAN21` at 1576
- `comfy/ldm/wan/` directory exists (only `model_wandancer.py` turned up in a shallow `find` —
  worth a full `ls` of that directory, the main model file is presumably named something else)
- **Confirmed.** `nodes.py:943` `UNETLoader.load_unet` calls:
  ```python
  unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
  model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
  ```
  `comfy.sd.load_diffusion_model(unet_path, model_options={})` (`comfy/sd.py:2053`) internally
  calls `comfy.utils.load_torch_file` + `load_diffusion_model_state_dict` (`comfy/sd.py:1956`,
  does the architecture auto-detection from the state dict) and returns a `ModelPatcher`. The
  actual `nn.Module` is expected at `model_patcher.model.diffusion_model` (ComfyUI's usual
  `ModelPatcher.model` = a `BaseModel` subclass, e.g. `WAN22`; `BaseModel.diffusion_model` = the
  wrapped transformer) — see `examples/loaders/wan_comfyui_loader.py` for the adapter built on
  this. Unconfirmed whether it needs a `.eval()` / dtype cast / device move before
  `torch.export.export` will trace it cleanly — that's the next thing to hit when actually
  running the export pipeline (Phase 2 in roadmap.md), not verified here.

**Update — actually ran on the RunPod instance:** `load_dit()` works as written. Result:

```
type: <class 'comfy.ldm.wan.model.WanModel'>
class: comfy.ldm.wan.model.WanModel
params: 14.29B
top-level children: ['patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'blocks', 'head', 'rope_embedder']
sample param dtype/device: torch.float32 cpu
```

- Confirms the class is `comfy.ldm.wan.model.WanModel` and the child module names match the
  tensor-name table above exactly, plus one addition: **`rope_embedder`** — a real submodule (no
  weight tensor shows up in the safetensors header for it since RoPE parameters are computed, not
  learned). Confirms `RotaryEmbedding` (plugins/csrc/rotary_embedding/) targets something real;
  its exact convention (rotate-half vs. interleaved) is still unverified against this module's
  actual `forward()` — check that before trusting the plugin's kernel math.
- **Loaded as `torch.float32` on `cpu`, not `float16`/`cuda`**, despite the checkpoint being fp16
  on disk. This is ComfyUI's own memory-management pattern: it keeps a full-precision master copy
  on the CPU "offload device" (used for e.g. LoRA merging) and only casts down to the compute
  dtype when it actually moves the model to GPU for a forward pass — `load_diffusion_model` alone
  never triggers that move. Practical implication for `--loader` / the export pipeline: don't
  assume the returned module is GPU-resident or fp16 — explicitly `.to("cuda")` and cast to the
  target export dtype before calling `torch.export.export`, or `export_to_torch_export` will
  trace against fp32-on-CPU tensors. Also means the module as returned is ~57GB in CPU RAM (fp32
  14.29B params), not the 28GB the file is on disk — fine on this pod (1.8TB RAM) but worth
  knowing before assuming disk-size-based memory budgets.

**Update — loader fixed to move to GPU/fp16 explicitly.** `load_dit()` now does
`.to(device="cuda", dtype=torch.float16)` (overridable via `TRTWAN_LOADER_DEVICE`/
`TRTWAN_LOADER_DTYPE`) before returning. Re-ran the same check afterward:

```
sample param dtype/device: torch.float16 cuda:0
```

Confirms the fix works — `load_dit()` now returns a GPU-resident, fp16 module ready to hand to
`torch.export`, not the fp32-on-CPU master copy `comfy.sd.load_diffusion_model` returns by
default.

**Update — low-noise expert checked too.** Ran the identical `load_dit()` check against
`wan2.2_i2v_low_noise_14B_fp16.safetensors`:

```
type: <class 'comfy.ldm.wan.model.WanModel'>
class: comfy.ldm.wan.model.WanModel
params: 14.29B
top-level children: ['patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'blocks', 'head', 'rope_embedder']
sample param dtype/device: torch.float16 cuda:0
```

**Identical to the high-noise expert** — same class, same 14.29B params, same children list, same
post-fix dtype/device. Confirms the two Wan 2.2 MoE experts are architecturally identical (only
the weight values differ), so a single `DiTExporter`/`exporter-kwargs` config works for both —
export/build each checkpoint separately (two engines), but no per-expert exporter changes needed.
Still doesn't resolve the MoE expert-switching gap in `DiTEngine`/`scheduler` (see above) — that's
about the *sampling loop* choosing between the two built engines mid-schedule, unrelated to
whether their export configs match.

## `forward()` signature — resolves the conditioning-mismatch question precisely

`WanModel.forward` (`comfy/ldm/wan/model.py`, via `inspect.signature`/`inspect.getsource` on the
loaded instance):

```python
def forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None,
            transformer_options={}, **kwargs):
    # dispatches through a WrapperExecutor to self._forward(...) with the same args
```

`_forward`'s first line is `bs, c, t, h, w = x.shape` — **`x` arrives already 36-channel.** There
is no separate `y`/mask/image argument anywhere in the signature. Whatever calls this model is
responsible for channel-concatenating noise-latent + image-latent + mask into a single `(B, 36,
T, H, W)` tensor *before* calling `forward` — this repo does not yet do that anywhere (see the
conditioning-mismatch section above; now precisely confirmed rather than inferred from weight
shapes alone). `clip_fea` is accepted but expected `None` for this checkpoint (no `img_emb.*`
weights loaded, consistent with the earlier finding). `time_dim_concat`/`context_latents`/
`reference_latent` are advanced features (temporal extension, in-context reference conditioning)
irrelevant to basic T2V/I2V export — safe to omit.

**Confirmed instance config** (constructor was called with non-default values; `__init__`'s
signature only showed class defaults):

| param | value |
|---|---|
| `model_type` | `'t2v'` — this checkpoint is task-flexible, not I2V-only: `in_dim=36` is fixed by the architecture regardless of task, and T2V generation is done by zero-filling the image-latent + mask portion of `x`'s 36 channels instead of populating them with a real reference frame. `model_type` is likely just ComfyUI's generic default label rather than a meaningful per-task flag — unconfirmed whether anything else in the codepath branches on it |
| `in_dim` | 36 |
| `out_dim` | 16 |
| `dim` | 5120 |
| `ffn_dim` | 13824 |
| `freq_dim` | 256 |
| `text_dim` | 4096 |
| `num_heads` | 40 (head_dim = 5120/40 = 128) |
| `num_layers` | 40 |
| `patch_size` | `(1, 2, 2)` |
| `text_len` | 512 |
| `qk_norm` / `cross_attn_norm` | both `True` |

All consistent with the tensor-shape-derived table earlier in this doc; `num_heads=40` and
`model_type` are new information from this pass.

**What this means for `DiTExporter`:** `example_inputs()` needs argument names matching
`forward()` exactly — `x` (not `latents`), `timestep`, `context` (not `text`) — and `x` must be
the full `(B, 36, T, H, W)` pre-concatenated tensor, `context` shaped `(B, <=512, 4096)`.

**Done:** `DiTExporter` (`export/exporters/dit.py`) now uses `x`/`timestep`/`context` and takes
`in_channels`/`text_dim` (renamed from `latent_channels`/`text_embed_dim`, which conflated the
DiT's input channel count with the VAE's unrelated 16-channel latent space). `DiTEngine`
(`engine/dit_engine.py`) now builds engine inputs keyed `x`/`context` to match, and
`_build_inputs` raises `NotImplementedError` for any non-text conditioning rather than silently
mis-routing it into a tensor name the engine doesn't have.

**Still open:** the *exact* channel order/construction of that 36-channel `x` (which 16 channels
are noise vs. image-latent, how the 4-channel mask is built) lives in whatever ComfyUI node
normally constructs it for this model (likely a stock `WanImageToVideo`-equivalent, not yet
located) — needed before a real I2V export, not needed for a T2V-only attempt (T2V could
plausibly zero-fill the extra 20 channels, unconfirmed). `example_inputs()` still just traces
against zeros for `x` — fine for proving export/ONNX/engine-build mechanics work, not sufficient
for a numerically correct I2V engine.

## `torch.export` succeeded — full model, real checkpoint

Fixed `load_dit()` to selectively cast only `patch_embedding` to fp32 (matching
`forward_orig`'s explicit `.float()` upcast for that one conv) while keeping the rest of the
14.29B-param model in fp16 — see the code comment in `wan_comfyui_loader.py` for why (blanket
fp32-casting the whole model OOM'd a 95GB GPU that already had another copy resident via the
running ComfyUI server; the targeted fix costs ~1.3M params' worth of extra memory instead of
14.29B).

With that fix, ran (tiny shapes, `B,C,T,H,W = 1,36,3,8,8`, `context` `(1,32,4096)`):

```python
exported = torch.export.export(m, args=(), kwargs={'x': x, 'timestep': timestep, 'context': context})
```

**Succeeded.** Eager forward also passed first (output shape `[1, 16, 3, 8, 8]` — confirms
`out_dim=16` as expected, and that padding/cropping round-trips correctly for this shape). The
exported graph:

- Traces cleanly through all 40 blocks, `patch_embedding`, `text_embedding`, `time_embedding`,
  `time_projection`, `rope_encode`, and the final `head` — no graph breaks, no data-dependent
  control flow that `torch.export` choked on, despite the conditional branches in `_forward` (the
  `time_dim_concat`/`context_latents`/`ref_conv` branches were simply not taken since we didn't
  pass those kwargs — untested whether those branches themselves are export-compatible, but
  irrelevant for a basic T2V/I2V export that never sets them).
- Preserves per-parameter dtype correctly: signature shows `p_patch_embedding_weight: "f32[5120,
  36, 1, 2, 2]"` while every block parameter is `"f16[...]"` — confirms the mixed-precision setup
  survives into the exported graph rather than getting silently unified to one dtype.

**This means the real blocker for a full export pipeline is no longer "does torch.export even
work" — it's wiring this up properly**: updating `DiTExporter.example_inputs()`/`input_names` to
use the real argument names (`x`, `timestep`, `context` — not `latents`/`text`) and the real
36-channel shape, and building the actual 36-channel `x` tensor (still not done — see the
conditioning-mismatch section above). Next step from here would be attempting `torch.onnx.export`
on this same `exported` program, and separately locating whatever ComfyUI code actually
constructs a real 36-channel `x` (image-latent + mask channel layout) for genuine I2V correctness.

## ONNX export: RMSNorm and RoPE both needed fixes

Attempted `torch.onnx.export(exported, ..., opset_version=20, dynamo=True)` on the program from
the section above. Two failures, in order, both now fixed:

**1. `aten._fused_rms_norm` had no ONNX translation at opset 20.** Wan uses RMSNorm for `norm_q`/
`norm_k` (`qk_norm=True`, confirmed in the config table above) throughout every block — not an
edge case, every real Wan export hits this. ONNX added a native `RMSNormalization` op at **opset
23**; PyTorch's dynamo ONNX exporter only maps the fused op to it at that opset or higher. Fixed
by bumping `ModelExporter.opset_version`'s default from 20 to 23 in `export/base.py` — a
framework-level fix, not a one-off script change, since this affects every DiT/text-encoder
export, not just this checkpoint.

**2. `comfy_kitchen.apply_rope1` had no ONNX translation at all** (it's not a standard ATen op,
so no opset bump helps). Traced to `comfy/ldm/flux/math.py`: `apply_rope1(x, freqs_cis)`
dispatches on `comfy.model_management.in_training` — `True` uses a pure-PyTorch fallback
(`_apply_rope1`, fully export-traceable), `False` (ComfyUI's normal inference default) uses the
opaque `comfy_kitchen` custom op. `comfy/ldm/wan/model.py` imports `apply_rope1` by name and
calls it directly on `q`/`k` (not through the paired `apply_rope` wrapper).

Fixed in `examples/loaders/wan_comfyui_loader.py`'s `load_dit()` by **cloning** (not depending
on) `_apply_rope1`'s exact math into this repo, then monkeypatching
`comfy.ldm.wan.model.apply_rope1` directly to point at the clone — deliberately not flipping the
broad `comfy.model_management.in_training` global, since its full effect on comfy's other
custom-kernel dispatch (attention, quantization ops — `comfy_kitchen`'s capability list includes
several fp8/nvfp4 quantize/dequantize/scaled_mm ops too) hasn't been audited, and this repo needs
to keep working outside a ComfyUI environment eventually — the RoPE math belongs owned here, not
borrowed at runtime from comfy internals that could change across releases.

**Important correctness finding surfaced while cloning this math:** Wan's actual RoPE convention
is **interleaved-pair rotation** — `_apply_rope1` reshapes `x` to `(..., -1, 1, 2)` (consecutive
element pairs `(x[2i], x[2i+1])` as a 2-vector) and rotates each pair by a 2x2 matrix built in
`rope()`. This is **not** the rotate-half convention (split the vector into first-half/
second-half chunks) that our own `RotaryEmbedding` TensorRT plugin
(`plugins/csrc/rotary_embedding/kernel.cu`) currently implements — see that file's own
docstring, which already flagged this as unvalidated. `comfy_kitchen`'s capability list
independently confirms the distinction exists as two separate kernels: `apply_rope1`
(interleaved, what Wan actually calls) vs. `apply_rope_split_half1` (rotate-half, unused by
Wan). **The plugin's kernel math needs rewriting to match the interleaved-pair convention before
it can be trusted** — `_apply_rope1` above (and `rope()` in `comfy/ldm/flux/math.py`) is the
reference to implement it against.

**Update — `torch.onnx.export` succeeded.** Re-ran with both fixes (opset 23 + the RoPE bypass,
initially via the broad `in_training=True` flag before it was narrowed to the surgical
`wan_model.apply_rope1` monkeypatch described above):

```
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX... ✅
```

Full ONNX conversion of the real 14.29B-param model succeeds end to end — `torch.export` +
`torch.onnx.export`, the first two of the three export pipeline stages, are now both proven
against real hardware. Only the TensorRT engine build (stage 3, `build_tensorrt_engine` /
`trtwan build engine`) remains unattempted.

**Output verified complete, not just "didn't error":** `dit_high_noise_test.onnx.data` is 26.62
GiB — matches 14.29B params × 2 bytes ÷ 1024³ ≈ 26.62 GiB almost exactly, confirming every
parameter actually made it into the external-data file (an earlier interrupted run produced a
suspiciously small 3.91 GiB file that looked like silent data loss; it wasn't — just a truncated
write from stopping the process early, not a real bug). Also confirms the narrowed
`wan_model.apply_rope1` monkeypatch (cloned `_apply_rope1`, not the broad `in_training` flag)
produces an identical successful result — that's the version to keep using going forward.

## How this was derived (re-runnable if the checkpoint is available again)

```bash
python -c "
import json, struct
path = 'wan2.2_i2v_high_noise_14B_fp16.safetensors'
with open(path, 'rb') as f:
    header_len = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_len))
header.pop('__metadata__', None)
for k in sorted(header):
    print(k, header[k]['shape'], header[k]['dtype'])
"
```

No `safetensors`/`torch`/GPU required — this only reads the JSON header, not the tensor data.

## TensorRT engine build (stage 3): environment + API findings

Two real issues hit attempting `build_tensorrt_engine`'s equivalent standalone
(`examples/loaders/test_trt_build.py`) against the tiny ONNX export from the section above, on a
RunPod instance with driver 570.195.03 (CUDA 12.8) and TensorRT 11.2.1.2.

**1. `pip install tensorrt` resolved the wrong CUDA runtime.** The plain `tensorrt` PyPI
metapackage pulled in `tensorrt_cu13` (`Requires: tensorrt_cu13` per `pip show`) — a CUDA 13
runtime — while the driver only supports up to CUDA 12.8 and every other installed NVIDIA package
(PyTorch's own deps) is cu12. Result: `trt.Builder(...)` raised `CUDA driver version is
insufficient for CUDA runtime version` / `TypeError: pybind11::init(): factory function returned
nullptr`. Fix: `pip uninstall tensorrt tensorrt_cu13` then `pip install tensorrt-cu12` — the
explicit cu12 variant, not the auto-resolving default. Worth checking `pip show tensorrt`'s
`Requires:` line on any fresh install rather than assuming the plain package name picks the right
CUDA runtime.

**2. `trt.BuilderFlag.FP16`/`BF16`/`FP8`/`INT8` do not exist in this TensorRT version's API for
`STRONGLY_TYPED` networks** (confirmed via `[f for f in dir(trt.BuilderFlag) if not
f.startswith('_')]` — none of the precision flags are in the list at all). This isn't a version
quirk to route around — it reflects a real design point: a `STRONGLY_TYPED` network's precision
is fixed entirely by the tensor dtypes already present in the ONNX graph (confirmed separately:
the parsed network's inputs/output were already `DataType.HALF` throughout, matching the fp16
export). There's no builder-level flag to opt into a precision after the fact anymore — the
"select precision" decision happens at export time (casting the model before `torch.export`, see
`wan_comfyui_loader.py`'s `.to(dtype=...)`), not at build time.

**Fixed in the framework, not just the script:** `export/trt_build.py`'s old
`_apply_precision_flags` (which called the now-nonexistent `config.set_flag(trt.BuilderFlag.FP16)`
etc. — would have crashed identically) is replaced with `_validate_precision`, which checks the
parsed network's actual input dtypes against the requested `PrecisionMode` and raises a clear
error on mismatch instead of either crashing on a missing API or silently building the wrong
precision. This also makes the FP8 gap (see roadmap.md) fail loudly rather than silently
succeeding with the wrong precision — a real behavior improvement, not just a compatibility fix.

ONNX parse itself succeeded cleanly once the cu12 tensorrt was installed: 15,020 layers, 3 inputs
(`x`, `timestep`, `context`) + 1 output (`noise_pred`), all `DataType.HALF` — confirms the earlier
export's dtype handling carried through correctly.

**Update — full engine build succeeded.** `builder.build_serialized_network(network, config)`
completed in 118.4s, producing a 26.63 GiB serialized engine (matches the 26.62 GiB ONNX weight
data almost exactly, as expected for an uncompressed fp16 strongly-typed build — no pruning or
compression happening, just repackaging into TensorRT's own format plus kernel selection
metadata). Saved to `dit_high_noise_test.engine`.

**This closes out the full three-stage export pipeline** — `torch.export` → ONNX → TensorRT
engine — proven end to end against the real 14.29B-param Wan 2.2 high-noise DiT, on real
Blackwell hardware (RTX PRO 6000). One caveat carried forward from every step before this: the
input shapes used throughout (`8×8`, 3 frames, static, no dynamic axes) are toy dummy values for
proving pipeline *mechanics* work, not a usable engine — no optimization profile was built (none
needed for a fully static network), and the 36-channel `x` input was all zeros rather than a real
channel-concatenated noise+image+mask tensor. A production build needs `DiTExporter`'s real
`dynamic_axes` wired through (already correct in the framework code, just not exercised by this
standalone script) and the still-open I2V conditioning construction from earlier in this doc.

## First real run through `trtwan export onnx` / `DiTExporter` found a genuine device bug

First time `DiTExporter` (not the hand-rolled standalone script) was actually exercised, via
`python -m tensorrt_wan.cli.main export onnx --component dit --loader
examples.loaders.wan_comfyui_loader:load_dit ...` with small-but-real dynamic shapes
(`latent_frames=3, latent_height=32, latent_width=32` — 32 is `DiTExporter.dynamic_axes()`'s
hardcoded min bound for height/width, so smaller values aren't valid without also patching that).

Failed immediately with `RuntimeError: Unhandled FakeTensor Device Propagation for aten.mm.default,
found two different devices cpu, cuda:0`, inside `time_embedding`'s first `Linear` layer.

**Root cause:** `ModelExporter.example_inputs()` implementations (all four: `DiTExporter`,
`TextEncoderExporter`, `VAEEncoderExporter`, `VAEDecoderExporter`) built tensors with plain
`torch.zeros(...)` — no `device=` argument, defaulting to CPU — while `load_dit()` moves the
model to `cuda`. This bug existed in the framework from the start; it was invisible until now
because every prior successful export in this doc went through `examples/loaders/test_export.py`,
a hand-written script that explicitly built its own `device='cuda'` tensors instead of calling
`DiTExporter.example_inputs()` at all. First real use of the actual exporter class immediately
found a bug three tests through a parallel path had been silently avoiding.

**Fixed:** added a `ModelExporter.device` property (`export/base.py`) that infers device from
`next(self.model.parameters()).device`, and updated all four `example_inputs()` implementations
to build their tensors with `device=self.device`.

## Second real finding: `Dim(min=, max=)` doesn't work for this model's spatial dims

Re-ran after the device fix — got past that cleanly, then hit a second, deeper issue:
`torch.fx.experimental.symbolic_shapes.ConstraintViolationError: Constraints violated`.

`export/torch_export.py`'s `_build_dynamic_shapes` was using `torch.export.Dim(name, min=axis.min,
max=axis.max)` for every dynamic axis — an *assertive* dynamic dim: torch.export requires the
traced code to behave consistently across the *entire* declared range, and fails hard if it can't
prove that. For `x`'s height/width axes (range `[32, 64]`, `patch_size=(1,2,2)`), the guard solver
concluded the range **specializes to a single value (32, exactly what the example input used)** —
`pad_to_patch_size`/`rope_encode`'s patch-alignment arithmetic (floor-division/modulo on the
spatial dims) generates guards that only hold for specific values, not smoothly across a range.
Batch (`x_dim0`) and frame-count (`x_dim2`) axes hit a related but distinct failure: "marked as
dynamic but your code specialized it to be a constant" — likely because unrelated code paths
(e.g. an internal broadcast/equality assumption) force those dims to match the example's concrete
value even though nothing about their own arithmetic *should* prevent them varying.

The error message suggested its own fix: *"If you're using Dim.DYNAMIC, replace it with either
Dim.STATIC or Dim.AUTO."* **Fixed** by switching `_build_dynamic_shapes` to `torch.export.Dim.AUTO`
for every axis instead of an explicit `Dim(min=, max=)` — `AUTO` lets torch.export *infer* what's
actually dynamic from the trace rather than us asserting a range upfront that conflicts with the
model's real constraints. `DynamicAxis.min/opt/max` are untouched by this change — they're
consumed separately, later, by `export.trt_build._build_optimization_profile` for the TensorRT
`IOptimizationProfile`, which never looks at torch.export's `Dim` objects at all. Not yet re-run
to confirm — next step. If `Dim.AUTO` also fails, the fallback is `Dim.STATIC` per-axis (giving up
on true dynamic shapes for this architecture and building one static engine per resolution
instead — which PLAN.md already lists as a supported strategy, not a fallback of last resort).

**Update — `Dim.AUTO` worked, and this was the first fully successful run through the real
framework code (`trtwan export onnx` CLI → `DiTExporter` → `export_to_torch_export` →
`export_to_onnx`), not the hand-rolled standalone script.** Both stages completed cleanly:

```
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX... ✅
Exported dit -> /workspace/runpod-slim/dit_high_noise_dynamic.onnx
```

Only warnings, no errors: `dimension inputs['x'].shape[0] 0/1 specialized; Dim.AUTO was specified
along with a sample input with hint = 1` for `x`/`timestep`/`context`'s batch dim (`dim0`) —
confirms batch genuinely can't be dynamic in this model's traced code (consistent with the
earlier hard failure under strict `Dim(min=,max=)`, now just handled gracefully instead of
crashing the whole export). **No equivalent warning for the frame-count/height/width axes** —
those stayed properly dynamic under `AUTO`, which is the part that actually matters for covering
multiple resolution profiles with one engine; a fixed batch size of 1 is a reasonable constraint
for video generation (one video per request) rather than a real limitation.

This closes the "real dynamic shapes" and "through the actual CLI/DiTExporter" items together —
both were blocked on the same two bugs (device mismatch, `Dim.AUTO` vs. `Dim(min=,max=)`), now
fixed in the framework itself.

## Third real finding: the earlier "successful" ONNX export was silently wrong

Ran `trtwan build engine` against `dit_high_noise_dynamic.onnx` (the ONNX from the section above).
TensorRT's ONNX parser rejected it:

```
[TRT] [E] IMatrixMultiplyLayer must have same input types. `A` is of type Float and `B` is of type Half.
[TRT] [W] IElementWiseLayer with inputs ONNXTRT_Broadcast_94_output and time_embedding.0.bias_output:
    first input has type Float but second input has type Half.
RuntimeError: Failed to parse ONNX model ...: In node 129 ... operator: Gemm (parseNode): INVALID_NODE
```

**Root cause: the earlier device-mismatch fix was incomplete.** `DiTExporter.example_inputs()`
got `device=self.device` added, but never `dtype=` — `torch.zeros(..., device=self.device)` with
no `dtype` defaults to **float32**. Because `patch_embedding` is *also* fp32 (the earlier
mixed-precision fix), `x.float()` inside `forward_orig` was a silent no-op — no crash there — but
the subsequent `.to(x.dtype)` then propagated `x`'s *actual* fp32 dtype (not the fp16 actually
intended) into everything downstream, mixing fp32 activations against fp16-weighted layers
throughout the rest of the graph. **`torch.export` and `torch.onnx.export` both tolerated this
silently** (PyTorch/ONNX allow implicit type mixing); only TensorRT's stricter parser caught it.
This means the earlier "`torch.onnx.export SUCCEEDED`" milestone produced a graph that was
technically valid ONNX but numerically wrong throughout most of the network — success at that
stage was necessary but not sufficient evidence of a correct export.

**Fixed properly this time:** added a `ModelExporter.dtype` property (`export/base.py`), fixed at
`torch.float16` — deliberately *not* inferred from `next(self.model.parameters()).dtype` the way
`device` is, since this model intentionally mixes precision (patch_embedding fp32, everything
else fp16) and "the first parameter's dtype" would be unreliable depending on iteration order.
Updated `DiTExporter`/`VAEEncoderExporter`/`VAEDecoderExporter`'s `example_inputs()` to pass
`dtype=self.dtype` on every floating-point tensor (`TextEncoderExporter`'s `input_ids`/
`attention_mask` are token-id integers, correctly unaffected). Not yet re-run — next step.

**Lesson for future work in this repo:** a successful `torch.export`/`torch.onnx.export` is not
proof of a *correct* export, only a structurally valid one — PyTorch's type system is more
permissive than TensorRT's. Don't treat ONNX export success alone as validation; the TensorRT
parse (or better, a numerical comparison against the eager PyTorch reference) is what actually
catches dtype-correctness bugs like this one.

## Fourth finding: declaring batch as a dynamic profile axis fails the TensorRT build

With the dtype fix in place, ONNX parsing succeeded cleanly (no more Gemm/dtype error). New
failure one step later, at `builder.build_serialized_network`:

```
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error (Dimension mismatch
for tensor x and profile 0. At dimension axis 0, profile has min=1, opt=1, max=4 but tensor
has 1.)
```

This is the same batch-specialization fact from the `Dim.AUTO` section above, showing up a second
time in a different tool: `Dim.AUTO` didn't just warn about specializing `x`/`timestep`/
`context`'s batch dim during `torch.export` — it actually **baked a fixed literal `1` into the
ONNX graph** for that dimension, not a symbolic/dynamic one. `DiTExporter.dynamic_axes()` was
still declaring a profile range (`min=1, opt=1, max=4`) for that same axis, and TensorRT correctly
refuses to set an optimization profile range on a dimension the network doesn't actually mark
dynamic.

**Fixed:** removed the `dim0` (batch) entries from `DiTExporter.dynamic_axes()` for `x` (and
removed `timestep`/`context` from the dict entirely, since their only would-be-dynamic dimension
was that same now-fixed batch — they're fully static in the graph, and `trt_build.py`'s
`_build_optimization_profile` skips any input name absent from `dynamic_axes()` rather than
needing an empty-list entry). Only `x`'s frame-count/height/width axes remain dynamic, which
matches what's actually true of the exported graph. No need to re-export the ONNX — the existing
file's graph already has this exact static/dynamic split baked in from the `Dim.AUTO` run above;
only the *profile-building* step needed to stop asking for something the graph doesn't have.

## Fifth finding: a reshape downstream of patch embedding may have a hardcoded token count

Re-ran after the batch-axis fix. Got past the profile-mismatch error, then hit two things in the
same run — a warning revealing a likely correctness gap, and a separate OOM that actually stopped
the build.

**Warning (didn't stop the build, but a real finding):**

```
[TRT] [W] Profile kMAX values are not self-consistent. IShuffleLayer node_view_11: reshaping
failed ... reshape would change volume 31457280 to 15728640 ... RESHAPE input dims{1 5120 6 32 32}
reshape dims{1 5120 3072}
[TRT] [W] Profile kMIN values are not self-consistent. IShuffleLayer node_view_11: reshaping
failed ... reshape would change volume 1310720 to 3932160 ... RESHAPE input dims{1 5120 1 16 16}
reshape dims{1 5120 768}
```

Decoded: at the profile's **min** shape (`T'=1,H'=16,W'=16` post-patchify → 256 tokens), the graph
tries to reshape into `768` tokens instead — and `768 = 3×16×16`, i.e. exactly the token count for
the **opt** shape (`latent_frames=3`), not the min shape actually being evaluated. Same story at
max: reshape target `3072` doesn't match `6×32×32=6144` (the actual max-shape token count) — `3072
= 3×32×32`, again matching `latent_frames=3` (the opt value) rather than the max shape's real
frame count. **A `view`/`reshape` operation downstream of `patch_embedding` appears to use a
token-count value baked from the `opt` example shape as a literal constant**, not a value derived
symbolically from the actual runtime input size — likely some intermediate Python-level shape
arithmetic in `forward_orig`/`rope_encode` that `Dim.AUTO` didn't manage to keep fully symbolic.

**Implication, not yet resolved:** even if a build succeeds, this suggests height/width (and
possibly frame-count) may not be *safely* dynamic for this architecture — inference at a shape
other than the opt one could reshape into a wrong-sized tensor rather than erroring cleanly. This
is real evidence in favor of the fallback already noted above (`Dim.STATIC` per resolution profile
— separate static engines per resolution, which PLAN.md already treats as a first-class supported
strategy) rather than continuing to chase full dynamic H/W/T support for this specific model.
Not investigated further yet — flagging rather than fixing, this needs its own dedicated pass.

**What actually stopped the build:** a separate, unrelated OOM —
`Requested amount of GPU memory (28579323904 bytes) could not be allocated` (~26.6GB, roughly the
model's own size). Root cause: `cli/commands/build.py`'s `run_engine` loads the full model via
`loader(checkpoint)` and keeps it GPU-resident for the entire function, including while
`build_tensorrt_engine` separately requests its own comparably-sized workspace for kernel
autotuning — two ~26-28GB allocations competing at once. Likely compounded by the ComfyUI server
also still running and holding memory (same root cause as the earlier OOM incident in this doc).

**Fixed:** `run_engine` now moves the loaded model to CPU (`model.to("cpu")` +
`torch.cuda.empty_cache()`) after constructing the exporter but before calling
`build_tensorrt_engine` — the build only reads `exporter.dynamic_axes()`/`example_inputs()` for
their *shapes* (not weight values) and parses `--onnx` from disk, it never needs the model's
actual weights resident on GPU. Not `del`'d entirely since `example_inputs()` still calls
`self.device`/`self.dtype`, which read `next(self.model.parameters())` and would break on a
`None` model — moving to CPU keeps that working while freeing the GPU memory.

**Update — build succeeded.** With the OOM fix (and the ComfyUI server process killed, removing
the other memory contender), `trtwan build engine` completed:

```
[TRT] [W] Profile kMAX values are not self-consistent ... (same reshape-constant warning as
    above — didn't block the build, see the finding above for what it means)
[TRT] [W] Profile kMIN values are not self-consistent ...
Built dit engine -> /root/.cache/tensorrt_wan/engines/e8d83be53d2ef25d.engine
```

**This closes the loop end to end**: `trtwan export onnx` → `trtwan build engine`, both through
the real CLI/`DiTExporter`/`build_tensorrt_engine`, with real dynamic shapes, fp16 precision
validated correctly, on the real 14.29B-param Wan 2.2 DiT, on real Blackwell hardware. Five real
framework bugs found and fixed along the way (device mismatch, `Dim.AUTO` vs. strict range, dtype
mismatch, batch-as-dynamic-axis, and the cache-directory issue below) — none of them script-local
hacks, all in the actual `tensorrt_wan` package.

## Sixth finding: engine cache defaulted to non-persistent storage

The built engine landed at `/root/.cache/tensorrt_wan/engines/...` — `CacheConfig.directory`'s
default (`~/.cache/tensorrt_wan/engines`) resolves to `/root/.cache` when running as root, which
is very likely *not* on this RunPod instance's persistent volume (`/workspace`), unlike
everything else in this project. A 26GB engine silently written somewhere that may not survive a
pod restart is a real risk, and there was **no CLI flag at all** to redirect it — every command
constructing `RuntimeManager()` used hardcoded defaults.

**Fixed:** added a global `--cache-dir` flag on `trtwan` itself (`cli/main.py`) and a shared
`cli/runtime_helpers.py:build_runtime(args)` helper that every command touching the cache
(`build`, `cache`, `inspect`, `optimization-report`, `gpu-report`) now goes through instead of
constructing `RuntimeManager()` directly — one flag, threaded everywhere, rather than duplicating
override logic per command. The engine already built was manually moved to
`/workspace/runpod-slim/trtwan_engines/` in the meantime.
