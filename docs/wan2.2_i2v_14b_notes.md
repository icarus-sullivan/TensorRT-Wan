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

**Update — `DiTEngine._build_inputs` now does the concatenation** (`engine/dit_engine.py`'s
`_concat_image_conditioning`, source-only change, no GPU here to run it): `first_frame`/
`last_frame` embeddings get channel-concatenated onto `x` in the confirmed
noise(16)++image_latent(16)++mask(4) order, zero-padded to the full temporal length with a binary
mask at the conditioned frame. `example_inputs()` still traces against zeros for `x` — that's
fine, tracing only needs correct shape/dtype, not real values (confirmed earlier: no
data-dependent control flow in the traced graph).

**Still open:** the *exact* padding/mask construction lives in whatever ComfyUI node normally
builds it for this model (likely a stock `WanImageToVideo`-equivalent, not yet located) —
`_concat_image_conditioning`'s zero-padding + binary-mask policy is a documented best-effort
default, not confirmed to match it (that node gray-pads pixel-space and VAE-encodes the whole
padded video, which likely produces non-zero latents for the padding frames — zero-padding here
may be numerically wrong even though shapes match). Needs locating that node's source and a
RunPod numeric comparison before trusting I2V output quality, not needed for a T2V-only attempt.

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

## 2026-08-06 session: re-verified DiT after the RoPE fix, then did text_encoder/VAE for the first time

Fresh RunPod instance (RTX PRO 6000 Blackwell, `torch.cuda.get_device_capability() == (12, 0)`,
i.e. SM120), `torch` 2.10.0+cu128 — notably newer than whatever version the sections above were
written against, and it surfaced two environment/version-skew bugs neither of which are logic
bugs in this repo's own code:

**`torch.export`'s `dynamic_shapes` dict validation got stricter.** `torch._dynamo.exc.UserError:
When dynamic_shapes is specified as a dict, its top-level keys must be the arg names [...] of
inputs, but here they are ['x']` — `export/torch_export.py`'s `_build_dynamic_shapes` only added
dict entries for inputs that actually had dynamic axes (`x` for `DiTExporter`), omitting
`timestep`/`context` entirely, which worked on whatever older torch version this was last run
against. Fixed: every `example_inputs()` key now gets an entry, `None` for inputs with no dynamic
axes, instead of being omitted.

**`torch.onnx`'s newer dynamo-based exporter needs `onnxscript`,** not bundled with `torch`
itself — `ModuleNotFoundError: No module named 'onnxscript'` on the first real `torch.onnx.export`
call. Fixed by `pip install onnxscript` on the pod; not a repo code change, just an environment
setup step worth recording since it wasn't needed on whatever torch version this repo's earlier
sessions used.

With both fixed, **re-ran the full DiT pipeline end to end and it still holds** with the RoPE
kernel fix from today (`plugins/csrc/rotary_embedding/`, see plugins.md) — `torch.export` → ONNX
(~44min, mostly the dynamo translation phase) → TensorRT engine (~35min, 26.6GiB, cache correctly
landing on `/workspace/runpod-slim/trtwan_engines/` via `--cache-dir`, not `/root/.cache`). Same
`kMAX`/`kMIN` "not self-consistent" reshape warning as before — already-tracked, didn't block the
build.

### Text encoder and VAE: first real attempt, new loaders written from scratch

No `load_text_encoder`/`load_vae_encoder`/`load_vae_decoder` existed before this session — only
`load_dit`. Added all three to `examples/loaders/wan_comfyui_loader.py`, deriving the real
attribute paths by loading each via ComfyUI's own code and introspecting on RunPod hardware
(same method `load_dit` was originally derived by):

- **Text encoder:** `comfy.sd.load_clip(..., clip_type=CLIPType.WAN)` → `.cond_stage_model` is a
  `WanTEModel`, whose `.umt5xxl` attribute (named after `WanT5Model`'s `name="umt5xxl"` kwarg) is
  an `SDClipModel` wrapping the real transformer at `.transformer` — a `comfy.text_encoders.t5.T5`
  instance whose `forward(input_ids, attention_mask, ...)` matches `TextEncoderExporter`'s input
  names directly. Its return is a `(x, intermediate)` tuple, not a single tensor — needed a thin
  `_TextEncoderWrapper` around it (`forward` returns `[0]` only) since `TextEncoderExporter`
  expects one `text_embeds` output.
- **VAE encoder/decoder:** `comfy.sd.VAE(sd=..., metadata=...).first_stage_model` is a
  `comfy.ldm.wan.vae2_2.WanVAE` (confirmed the real dispatch target for `wan2.2_vae.safetensors`
  specifically, not the Wan 2.1 `comfy.ldm.wan.vae.WanVAE` class) — has separate `.encode(x)`/
  `.decode(z)` methods, not a unified `forward`, so needed two thin wrappers
  (`_VAEEncodeWrapper`/`_VAEDecodeWrapper`) each exposing one direction as `forward`.
  **Not yet investigated:** both `WanVAE.encode`/`.decode` have a Python-level `for i in
  range(iter_)` loop where `iter_` is computed from the input's actual temporal size (chunked
  causal-conv processing, 4 frames per chunk) — this is data-dependent control flow, the same
  category of thing that broke `Dim(min=,max=)` on the DiT months ago. Neither export was
  actually attempted this session (see below) so whether `Dim.AUTO` handles this by specializing
  cleanly (like DiT's batch dim did) or hard-fails is still unknown — flagging rather than
  chasing further tonight, this needs its own dedicated pass before VAE export is attempted.

**Text encoder ONNX export succeeded first try** (~5s, batch specialized to 1 same as DiT, no
other issues) — `text_encoder.onnx` (2.7MB graph + 11.3GB external weights).

**Text encoder engine build found two more real bugs, then hit a real dead end:**

1. `export/trt_build.py`'s `_validate_precision` rejected `input_ids`/`attention_mask` for being
   `INT64` instead of the requested `fp16` — a real bug, not a text-encoder-specific one: the
   check was written against the DiT, where every input genuinely is a float activation, and
   never accounted for legitimately-integer inputs (token ids, an int/bool mask) that were never
   supposed to match a float precision in the first place. **Fixed:** only floating-point-typed
   inputs are checked now; int/bool inputs are skipped entirely.
2. Same batch-as-dynamic-profile-axis bug the DiT hit before (`docs` section above, "Fourth
   finding") — `TextEncoderExporter.dynamic_axes()` still declared a `dim0` (batch) profile range
   despite the graph specializing batch to a fixed value (confirmed by the export-time "0/1
   specialized" warning). **Fixed:** removed `dim0` from `TextEncoderExporter.dynamic_axes()`,
   same fix pattern as `DiTExporter`'s.
3. **Real dead end, not fixed:** past both of those, the build failed with `MyelinCheckException:
   ... Attention operation was not supported by a dedicated kernel`, for T5's masked
   self-attention specifically (an additive float mask built from `attention_mask`, not a
   causal/simple-padding pattern). The error names its own suggested fix —
   `IAttention::setDecomposable` — which exists in the C++ API (`trt.IAttention.decomposable` is
   a real property, confirmed via `dir()`), but **is not reachable from Python in this TensorRT
   version (11.2.1.2)**: `network.get_layer(i)` returns the generic base `ILayer` for
   `ATTENTION_INPUT`/`ATTENTION_OUTPUT` layer types rather than auto-downcasting to
   `IAttentionInputLayer`/`IAttentionOutputLayer` the way older/more established layer types do,
   and there's no Python-callable cast — confirmed directly by testing both `layer.attention`
   (`AttributeError`) and `trt.IAttentionInputLayer(layer)` (`TypeError: No constructor
   defined!`) against the real parsed network (48 attention-boundary layers found, same failure
   on all of them). Checked NVIDIA's fused-attention-kernel docs
   (`transformers-fused-attention.html`): for our GPU's bucket (SM75-SM90/SM120/SM121), FP16 head
   dim 16–256 and "any masking" are both claimed supported, and the DiT's own (unmasked)
   attention did find a dedicated kernel with no such error — so this looks like a real gap
   between that doc and TensorRT 11.2's actual native-ONNX-`Attention`-op import path
   specifically for a masked case, not a config mistake on this repo's side.

**Update — the real fix was attempted and worked, for both text encoder and VAE.** Added
`_decomposed_sdpa`/`_decompose_attention_for_export` to `wan_comfyui_loader.py`: monkeypatches
`torch.nn.functional.scaled_dot_product_attention` (the function every comfy attention path
bottoms out in, confirmed by reading `comfy/ops.py`) to a manual matmul+softmax+matmul
decomposition — same technique as `_apply_rope1`'s RoPE monkeypatch, applied only in
`load_text_encoder`/`_load_wan_vae`, not `load_dit` (whose attention already finds a fused kernel
and doesn't need it). This turned out **not to be mask-specific**: the VAE's bottleneck
self-attention (unmasked) hit the exact same `MyelinCheckException` the masked text-encoder
attention did, so the native opset-23 `Attention` op import path itself is the problem in this
TensorRT version, not anything about the mask. After re-exporting with the decomposed form, both
components built cleanly. Text encoder engine build took ~28min (vs. VAE's ~1min) — decomposing
costs real build/runtime time relative to a fused kernel, but it's now *correct*, not broken.

**Second real export bug, found building the VAE (not the text encoder):**
`torch._subclasses.fake_tensor.UnsupportedOperatorException: aten.cudnn_convolution.default`.
Root cause: `comfy.ops.Conv3d._conv_forward` calls `torch.cudnn_convolution` directly (bypassing
`nn.Conv3d`'s normal path) whenever `comfy.ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND` is `True` — a
real, deliberate workaround for an actual cuDNN memory bug, gated on cuDNN 9.10.2–9.15.0 + torch
2.9–2.10 (`comfy/ops.py`), which matches this environment's versions exactly. `cudnn_convolution`
has no FakeTensor/meta kernel, so `torch.export` can't trace through it. **Fixed** by monkeypatching
`comfy.ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND = False` in `_load_wan_vae`, safe because
non-strict `torch.export` never runs a real cuDNN kernel during tracing (FakeTensors only) — the
actual memory bug this workaround exists for is simply not in play at export time, and this never
touches a running ComfyUI server (separate process).

**VAE encoder export succeeded** once both of the above were fixed — and confirmed the
data-dependent-shape question flagged above: the export-time warning showed `pixels.shape[2]`
(the frame axis) specializing to a fixed value, same "0/1 specialized" pattern as DiT's batch dim,
*not* a hard failure. `Dim.AUTO` handled `WanVAE.encode`'s chunked-loop trip count by baking in
the traced `frames` value rather than erroring — confirms this export is only valid for the
traced frame count (T=1 for single-image/I2V-reference-frame use), not genuinely T-dynamic.
Fixed `VAEEncoderExporter.dynamic_axes()` to drop both `dim0` (batch) and `dim2` (frames),
keeping only height/width dynamic, mirroring the same fix pattern already applied twice.

**Third real finding, and an important correction: wrong VAE checkpoint file.** VAE decoder
export hit `RuntimeError: Given groups=1, weight of size [48, 48, 1, 1, 1] ... but got 16
channels` — traced by directly introspecting the loaded model (`vae.first_stage_model.z_dim`)
rather than guessing from source: `wan2.2_vae.safetensors` dispatches to
`comfy.ldm.wan.vae2_2.WanVAE` with **`z_dim=48`**, not 16. Checked `wan_2.1_vae.safetensors`
the same way: dispatches to the older `comfy.ldm.wan.vae.WanVAE` class, **`z_dim=16`** — matching
the DiT's expected 16-channel noise/image-latent portions exactly. Conclusion: Wan 2.2 ships a
*separate*, newer 48-channel VAE (`wan2.2_vae.safetensors`) for its 5B TI2V model specifically;
the 14B I2V MoE checkpoints this repo targets (`wan2.2_i2v_{high,low}_noise_14B_fp16`) still pair
with the original Wan 2.1 16-channel VAE. **`wan2.2_vae.safetensors` was the wrong file for this
repo's loader to use against these checkpoints** — `load_vae_encoder`/`load_vae_decoder`'s
docstrings/usage should point at `wan_2.1_vae.safetensors`, not the file its name would suggest.
The VAE encoder build from the previous section was against the *wrong* checkpoint (48ch,
incompatible with the DiT) and was rebuilt against the correct one after this was found.

**Fourth real finding, unrelated to the VAE itself but found while rebuilding:** `EngineCache`'s
`CacheKey` (`runtime/cache.py`) had no `component` field — `vae_encoder` and `vae_decoder` share
one checkpoint file (identical `model_hash`) and were built with the same profile/precision too,
so they collided on the exact same cache digest. Confirmed as a live bug, not theoretical: the
first (mis-targeted, wrong-checkpoint) `vae_decoder` build attempt returned "Using cached engine"
and silently served back the *encoder's* engine file. **Fixed:** added `component: str` to
`CacheKey`, populated from `exporter.name` at both construction sites (`cli/commands/build.py`,
`export/pipeline.py`); old cache entries (missing the field) safely stop matching rather than
crashing, per `EngineCache.get()`'s existing "metadata mismatch; treating as a miss" handling —
no manual cache cleanup needed. Added a regression test
(`test_different_component_is_a_different_cache_entry`) to `tests/test_cache.py`.

**Both VAE encoder and VAE decoder engines then built successfully** against the correct
`wan_2.1_vae.safetensors` checkpoint. `VAEDecoderExporter.dynamic_axes()` got the same
batch+frame-axis fix as the encoder, by analogy (not independently re-confirmed via its own
specialization warning, but the export succeeded with it applied).

**All four components (DiT, text_encoder, vae_encoder, vae_decoder) have built TensorRT
engines**, all in `/workspace/runpod-slim/trtwan_engines/`.

## First real end-to-end I2V run: two serious infra bugs found and fixed, generation quality still open

Wrote a standalone integration script (`/workspace/runpod-slim/i2v_smoke_test.py`, scratch, not
committed) driving all four engines directly — real prompt ("the camera moves to the left as it
zooms out to show the full green chair"), two real reference images
(`close_green_chair_{start,end}.png`), real ComfyUI tokenizer (not `transformers.AutoTokenizer`,
which would use the wrong vocab entirely), `FlowMatchEulerScheduler`. Resolution pinned to
256x256 (32x32 latent) — exactly the DiT engine's opt shape, sidestepping the known
reshape-constant-baking bug rather than risking a wrong-but-not-crashing result at a non-opt
shape. Latent frame count 3, matching both DiT's opt and the VAE decoder's fixed frame axis.

**Zeroth bug, caught before ever running:** the first version of `DiTEngine._build_inputs`
concatenated image conditioning once *per kind* — correct for `first_frame` alone, but
`first_frame` + `last_frame` together would have produced 56 channels against the engine's fixed
36. **Fixed** before any GPU time was spent on it: `_concat_image_conditioning` now builds one
combined 16-channel image_latent + one combined 4-channel mask across every present kind, each
placed at its own temporal position, matching the confirmed single-slot architecture.

**GPU memory contention, not a bug:** first run OOM'd loading the DiT engine.
`nvidia-smi --query-compute-apps` showed a concurrent ComfyUI server (the user's own session)
holding ~66GB of the 97.9GB GPU. Rewrote the script to load/use/unload engines one at a time
(peak ~28.6GB) as a `SEQUENTIAL` toggle — worth keeping for low-VRAM setups generally, not just
this incident. Once the user confirmed the ComfyUI server was done and it was terminated
(SIGTERM, GPU fully freed to 0MiB used), switched to loading all four upfront instead.

**First real crash-free run produced a wall of TensorRT `setInputShape` errors, silently
ignored.** `TensorRTEngineWrapper._infer_trt` never checked `context.set_input_shape()`'s return
value — it returns a bool rather than raising. Root cause: `SchedulerState.current_timestep`
(`scheduler/state.py`) returns a 0-d scalar (`self.timesteps[self.step_index]`, indexing a 1-d
tensor with a plain int), but `DiTExporter.example_inputs()` declared `timestep` as rank-1
(shape `(1,)`) at export time — a real rank mismatch on every single DiT call, `nbDims` 0 vs 1.
**Fixed two things:** (1) `DiTEngine._build_inputs` now does `timestep.reshape(1)` (a no-op if
already rank-1); (2) `TensorRTEngineWrapper._infer_trt` now checks `set_input_shape`'s return
value and raises loudly instead of silently continuing with a stale binding — this class of bug
(TensorRT API returning failure via bool instead of exception) could bite any future input, not
just this one.

**Re-ran clean of the shape error, but `final_latents` came out all-NaN — again traced to the
same `timestep` tensor, a different aspect of it.** `cond_out` was already NaN on the *first*
denoising step (timestep=1000.0), before any accumulation. Root cause:
`TensorRTEngineWrapper._infer_trt` calls `context.set_tensor_address(name, tensor.data_ptr())`
with **no dtype conversion** — `FlowMatchEulerScheduler`'s `timestep` is float32
(`torch.linspace`'s default dtype), but the DiT engine's `timestep` binding was exported as
float16. TensorRT reinterpreted the same raw bytes as fp16 instead of converting them, producing
garbage. **Fixed generically, not just for `timestep`:** `_infer_trt` now casts every input
tensor to `_trt_dtype_to_torch(engine.get_tensor_dtype(name))` before use — protects every input
on every engine from this exact silent-corruption class of bug, not a narrow timestep-only patch.
(Along the way, `_trt_dtype_to_torch`'s mapping was also missing `INT64` entirely — added it;
needed for `text_encoder`'s legitimately-int64 `input_ids`/`attention_mask`, which broke when the
dtype-cast fix first landed and immediately surfaced the gap.)

**With both fixes, the pipeline runs numerically clean end to end — no NaN, no crashes, sane
value ranges at every stage** (confirmed via a `stats()` helper printing min/max/mean/std/nan/inf
at each stage: `first_frame_latent`/`last_frame_latent` after VAE encode, a **VAE round-trip
diagnostic** — decoding `first_frame_latent` alone (repeated to fill 3 frames) completely
bypassing DiT — which produced a genuinely recognizable image of the green chair, confirming VAE
encode/decode and the earlier `wan_2.1_vae.safetensors` checkpoint correction are both actually
correct, not just shape-plausible), `initial_latents`, per-step `latents` during denoising, and
final decoded pixels.

**But the actual generated *content* is wrong: a flat, near-uniform color, not the chair.** Ran a
systematic 2x2 isolation (guidance_scale 5.0 vs 1.0, real vs zeroed-out image conditioning) to
localize the remaining bug:

| guidance_scale | image conditioning | final_latents (mean / std) | visual result |
|---|---|---|---|
| 5.0 | real (both frames) | 1.76 / 4.79 | flat olive-green blob |
| 1.0 | real (both frames) | 0.56 / 1.59 | high-frequency static/noise, unconverged |
| 5.0 | zeroed | 1.33 / 4.96 | flat blob, same pattern as real |
| 1.0 | zeroed | 0.53 / 1.72 | static/noise, same pattern as real |

**Clean conclusion: image-conditioning content has negligible effect on the failure pattern** —
zeroing it out changes the result by noise-level amounts, not qualitatively. This rules out
`_concat_image_conditioning`'s mask/padding policy (the piece flagged as "best-effort,
unconfirmed" since it was first written) as the primary suspect, at least for this failure mode.
The failure instead tracks `guidance_scale` cleanly: high CFG diverges to flat saturation, CFG=1
(single conditional pass only, no combination at all) still fails to converge to a clean image,
just in a different way (looks like insufficiently-denoised noise). Since CFG=1 uses *only* the
conditional forward pass — no `uncond`/combination math involved at all — this points at
something in the single-pass DiT conditional output itself not being quite right, independent of
the CFG formula.

**Verified NOT the bug, to narrow the search:**
- The CFG combination formula (`uncond_out + guidance_scale*(cond_out-uncond_out)` on raw
  velocity, done in `DiTEngine.denoise_step`) is mathematically equivalent to combining on
  `calculate_denoised`-style x0 estimates instead (verified algebraically against ComfyUI's real
  `CONST.calculate_denoised` — `model_input - model_output*sigma` — since that transform is
  linear in `model_output`, combining before or after it commutes). Not a CFG-formula bug.
  Not the image-conditioning content, per the table above.
- `FlowMatchEulerScheduler`'s sigma schedule (`shift*sigma/(1+(shift-1)*sigma)` over a linear
  `[1,0]` ramp) matches ComfyUI's real `time_snr_shift`/`ModelSamplingDiscreteFlow.sigma` formula
  exactly (`comfy/model_sampling.py`) — checked directly against source, not assumed. `shift=5.0`
  specifically isn't confirmed as Wan's exact default (not statically hardcoded anywhere found in
  `supported_models.py`; workflow-configurable in practice), so it's *a* plausible remaining
  variable, just not a formula-level bug.

## Shift sweep and the decisive eager-vs-TensorRT comparison

**`shift` ruled out too.** Swept `shift` in `[1.0, 2.0, 3.0, 5.0, 8.0]` (real image conditioning,
`guidance_scale=1.0`, same prompt/images, one engine-load session reused across all five —
`/workspace/runpod-slim/i2v_shift_sweep.py`, scratch). Every value converged to nearly identical
`final_latents` stats (mean 0.53–0.57, std 1.50–1.61 — a tight cluster) and visually identical
high-frequency noise/static output. `shift` is not the cause. Each full run (20 steps + decode)
took only 2–7 seconds at this resolution — engine loading (~3.5min for all four, ~49GB) was the
actual dominant cost, not generation itself.

**The decisive check: does the built DiT TensorRT engine match the real eager PyTorch model?**
Two-phase script (`save_conditioning.py` + `compare_eager_vs_trt.py`, scratch): phase 1 used the
already-built text_encoder/vae_encoder engines to compute and save real `text_embeds`,
`first_frame_latent`, `last_frame_latent`, and a fixed-seed `initial_latents` to disk. Phase 2
built `x` (36-channel: noise++image_latent++mask, identical construction to
`DiTEngine._concat_image_conditioning`) at `timestep=1000.0` (the first denoising step — where
the earlier NaN investigation focused), then ran **both** the DiT TensorRT engine and
`load_dit()`'s real eager 14.29B-param checkpoint on the exact same input tensors and compared:

```
max_abs_diff=0.0137   rel_err=0.35%   cosine_similarity=0.999995
```

**This is essentially a perfect match** — well within normal fp16 numerical noise, not a
divergence. **Conclusively rules out the export/build pipeline as the source of the content-quality
bug**: the RoPE interleaved-pair fix, decomposed attention (avoiding TensorRT's native-Attention-op
dead end), and the rest of the export chain are all numerically verified correct — the built
DiT engine faithfully reproduces the real checkpoint's computation for this input.

**This flips the investigation.** Since eager and TensorRT agree almost exactly on this exact
input, and that input still produces incoherent output end-to-end, the remaining bug cannot be in
engine conversion — it has to be in what gets fed to the model. Both eager and TensorRT would
be equally "confused" by a semantically-wrong input and produce equally wrong (but
mutually-consistent, as observed) output. The prime remaining suspect is exactly the piece
flagged as unconfirmed since `_concat_image_conditioning` was first written: the real
`WanImageToVideo` node's image_latent/mask construction (pixel-space gray-fill before a full-video
VAE encode, real per-channel mask semantics) was never located or read in this environment — this
repo's zero-padding + binary-mask policy is a plausible-shaped guess, not a confirmed match, and
is now the single most likely remaining explanation for bad generation quality.

## Found and fixed: real channel-order bug — first genuinely coherent output

Located the real node: `comfy_extras/nodes_wan.py`'s `WanImageToVideo` (the official/canonical
one — `grep -rl "class WanImageToVideo"` also turned up two custom-node forks, ignored). Its
`execute()` builds the mask/image-latent conditioning values but does **not** itself
channel-concatenate them onto the DiT's `x` — that happens later, in `WAN21.concat_cond`
(`comfy/model_base.py`), which is what actually matters for matching this repo's
`_concat_image_conditioning`.

**Real channel order, read directly from `concat_cond`:**
```python
mask = 1.0 - mask  # WanImageToVideo produces 0=known/1=to-generate; this inverts it
...
return torch.cat((mask, image), dim=1)  # then concatenated after noise: noise ++ mask ++ image
```
i.e. **`noise(16) ++ mask(4) ++ image_latent(16)`** — mask *before* image latent.
`_concat_image_conditioning` had `noise ++ image_latent ++ mask` (image before mask) since it was
first written, always flagged as an unconfirmed guess. **This was the real bug**: with the order
reversed, the model's mask-weight-slice and image-latent-weight-slice were being fed entirely
each other's data — structurally consistent with every generation attempt producing zero
coherent structure regardless of CFG, shift, or image content, exactly the "garbage in" signature
the eager-vs-TensorRT comparison pointed at (engine verified correct; input to the engine wasn't).

Mask *polarity*, traced through both `WanImageToVideo` and `concat_cond`'s inversion, nets out to
1=known/0=to-generate — which is what this repo already implemented. Not a bug, despite looking
backwards at a glance against the node alone.

Still confirmed-but-unfixed: `WanImageToVideo` gray-fills (pixel value 0.5) every frame without a
real reference *before* VAE-encoding the whole padded video in one call — so its "padding" latent
frames are whatever the VAE produces for gray input, not zero. `_concat_image_conditioning` still
zero-pads directly in latent space. Smaller-magnitude than the channel-order bug, not yet fixed
or measured.

**Fixed** (`engine/dit_engine.py`'s `_concat_image_conditioning`): swapped concatenation order to
`torch.cat([x, mask, full_image_latent], dim=1)`.

**Re-ran the real prompt/images end to end (guidance_scale=5.0, real image conditioning) — first
genuinely coherent output of the whole project.** `final_latents` stats transformed completely:
mean=0.05, std=1.10 (vs. mean=1.76, std=4.79 with the channel-order bug) — stable, well-behaved
values throughout all 20 steps rather than runaway drift. Decoded frames show real spatial
structure for the first time: a dark rectangular band in the chair's position/shape against a
lighter wall background with vertical line detail (matching the window/pipe visible in the
source photos), consistent across all 9 output frames, not a one-off. Still low quality (20
steps, 256×256 test resolution, unfixed gray-fill-padding discrepancy above, mask/frame-index
semantics for `first_frame`/`last_frame` still not independently confirmed beyond "the obvious
reading of the kind's name") — but structurally working for the first time, a fundamentally
different result from every flat-blob/static-noise attempt before it.

**Not yet done:** higher step count / full resolution / quality comparison against a real
ComfyUI-generated reference video; the gray-fill-padding fix; independent confirmation of the
first_frame=index-0/last_frame=index-(-1) temporal convention.

## 2026-08-06 (second session): gray-fill fix, fresh-pod setup, and a real fp8 mislabeling bug

Picked up on a **new RunPod instance** (fresh container — no TensorRT-Wan checkout, no TensorRT
installed, no bash history; only `/workspace` persisted as a network volume with a pre-existing
ComfyUI install and models). Confirms `/workspace` is the persistent boundary on this box — a repo
checkout or built engines placed outside it (e.g. under `/root`) do not survive a pod restart, on
top of the already-known `~/.cache` pitfall from the first session.

### Gray-fill-padding fix, ported from real ComfyUI source

Fetched `comfy_extras/nodes_wan.py` directly from `github.com/comfyanonymous/ComfyUI` (no local
clone available on the Mac dev machine) and read `WanFirstLastFrameToVideo.execute()` verbatim —
the node that handles both first- and last-frame conditioning together (`WanImageToVideo` alone
only takes a single `start_image`, no end frame). Confirmed exactly:

```python
image = torch.ones((length, height, width, 3)) * 0.5
mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
if start_image is not None:
    image[:start_image.shape[0]] = start_image
    mask[:, :, :start_image.shape[0] + 3] = 0.0
if end_image is not None:
    image[-end_image.shape[0]:] = end_image
    mask[:, :, -end_image.shape[0]:] = 0.0
concat_latent_image = vae.encode(image[:, :, :, :3])
mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
```

Real algorithm: gray-fill (pixel 0.5) the *entire* target-length video, overwrite real first/last
frames, **one `vae.encode()` call over the whole padded video** (not per-frame) — so padding
latent frames reflect the VAE's real causal-conv response to gray input next to real frames, not
exactly zero. The mask is built at raw-pixel-frame granularity (`latent_frames * 4`) then reshaped
into 4 channels — this produces a real, non-obvious **asymmetry**: a single first frame marks all
4 mask channels known at latent index 0 (`:1+3` spans a whole causal group), but a single last
frame marks only 1-of-4 channels known at the last latent index (`-1:` is just the raw frame's own
slot). Also independently re-confirms `first_frame=index 0 / last_frame=index -1` directly from
`image[:start_image.shape[0]] = start_image` / `image[-end_image.shape[0]:] = end_image`.

Ported this exactly into `api/wan_engine.py`'s new `_build_image_to_video_conditioning`, called
from `WanEngine.generate()` (which now also takes `last_image=`, previously unwired in the
standalone API even though `dit_engine.py` already supported `LAST_FRAME` structurally). Landed as
a **separate code path** from the ComfyUI-graph one, not a rewrite of
`_concat_image_conditioning`: that function encodes one frame at a time
(`comfyui/nodes/vae_encoder.py`'s `TensorRTVAEEncoder`, one node per frame) with no visibility into
the target video's full length, so it structurally can't build the real padded-video-encode — it
keeps the old zero-pad approximation. `dit_engine.py` gained a `ConditioningKind.IMAGE_VIDEO` fast
path (`_PREBUILT_IMAGE_VIDEO_KEY`) that concatenates the standalone path's already-complete
image_latent/mask directly, skipping placement math. CPU-only tests added
(`tests/test_image_conditioning.py`) verifying the asymmetric mask and both dit_engine branches
against fake stand-ins — no GPU/model needed, ran clean both locally and on the pod (39 tests,
1 pre-existing unrelated failure in `test_config.py`'s yaml round-trip).

**Consequence for engine builds:** `_build_image_to_video_conditioning` calls
`vae_encoder.encode_video()` with the *full* `num_frames` (81 for the default profile), not
`T=1`. `VAEEncoderExporter`'s data-dependent chunked-loop trip count (documented in its own
`dynamic_axes()` docstring) means an engine built with the default `frames=1` kwarg is **only**
valid for single-image encode — it will not serve `encode_video` at T=81. The vae_encoder engine
must be built with `--exporter-kwargs '{"latent_channels": 16, "frames": 81}'` for this fix to
work at all. (The `T=1` engine's `encode_image` path is now dead code in `WanEngine.generate()`
either way, since `ConditioningKind.IMAGE`'s registration was never exercised by `generate()`.)

Also caught and fixed a **stale-doc landmine** in `wan_comfyui_loader.py`'s `_load_wan_vae`
docstring: it still said to use `models/vae/wan2.2_vae.safetensors`, which the *first* session
already found and fixed as the wrong file (z_dim=48, for the separate 5B TI2V model) —
`wan_2.1_vae.safetensors` (z_dim=16) is correct for these 14B checkpoints. The doc had drifted out
of sync with the fix. Would have silently reintroduced that exact bug on a fresh build if not
caught before running it.

### Real bug: `--precision auto` silently mislabels an fp16 build as "fp8" on Blackwell

Ran the build pipeline (`build_all.sh`: text_encoder -> vae_encoder -> vae_decoder -> dit,
targeting the `480x832`/81-frame default profile, `--precision auto` per the CLI's own default).
`text_encoder`'s build logged `Precision selected: fp8 (auto: blackwell (sm_120) default
ceiling)` and **succeeded**, producing a 20.6GB engine and caching it labeled `precision: "fp8"`.

That's wrong on two counts, both real, both worth fixing eventually (not fixed this session —
out of scope for today's task, flagged in docs/roadmap.md instead):

1. **`runtime/precision.py`'s `select_precision` has no actual FP8 quality gate.** Its own
   docstring promises "Blackwell -> FP8 where a per-op quality check clears it, FP16 otherwise" —
   the code just unconditionally returns the architecture's max-precision ceiling (`"fp8"` on
   Blackwell) whenever `config.allow_fp8` is true. No per-op check exists anywhere. This matches
   an already-tracked roadmap item ("Per-op FP8 quality gating on Blackwell") — this session found
   a concrete instance of it actually firing, not just a theoretical gap.
2. **`export/trt_build.py`'s `_validate_precision` only checks *network input* tensors, not
   internal weights.** The loader always casts the model to fp16 regardless of requested
   precision (never reads `precision` at all) — so a "fp8" build should be catching a real
   fp16-vs-fp8 dtype mismatch and failing loudly, exactly as its own docstring promises ("fails
   loudly instead of silently building the wrong-precision engine"). It does — for components
   whose *inputs* are float (dit's `x`/`context`, vae's `pixels`/`latent`). But `text_encoder`'s
   inputs are `input_ids`/`attention_mask`, both `INT64` — zero float tensors to check, so the
   loop silently finds nothing wrong and the "fp8"-labeled, actually-fp16 engine builds clean.
   Would very likely have hard-failed on the very next step (`vae_encoder`, real float `pixels`
   input) — and almost certainly on `dit` (real float `x`/`context` inputs) after burning the most
   GPU time of the whole pipeline getting there. Caught this by noticing the 20.6GB text_encoder
   engine size (fp16 UMT5-XXL-sized, not fp8-sized) and cross-checking the log line before letting
   the background build proceed further — stopped it (`TaskStop`, then had to `kill -9` the
   detached remote process directly since closing the local ssh connection alone didn't stop it —
   confirmed via `ps aux` on the pod), then reran the whole pipeline with `--precision fp16`
   passed explicitly on every `build engine` call.

**Takeaway for next time:** on this Blackwell box, always pass `--precision fp16` explicitly.
Never rely on `--precision auto` (the CLI's own default) until `select_precision`'s promised
per-op quality gate and a real PTQ/calibration pass actually exist — until then `auto` on
Blackwell is not a "safe default that falls back correctly," it's a silent mislabel that gets
caught by luck (a component with float inputs) or not (one without).

### `vae_encoder` fails to build at T=21/T=81 — boundary vs. last night's known-good T=9 still open

With `--precision fp16` fixed, the pipeline got past `text_encoder` clean, then **`vae_encoder`
failed at the TensorRT build step** (`frames=81`, needed for the gray-fill fix's single
whole-video `encode_video()` call):

```
[TRT] [E] Error Code: 9: ... Autotuner: failed to find fallback kernel:
  ...: fc: __mye...-(f16[2,__mye..._proxy.1,__mye..._proxy.1][]...) | ...(f16[2,...,384][]...), 
  ...(f16[2,384,...][]...), ...node_matmul_20_alpha..., ...node_matmul_20_beta...
  // node_matmul_20 fusion: cask In compileGraph ...
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error 
  (Could not find any implementation for node {ForeignNode[node_slice_2083...node_Split_34622]}.
  In computeCosts ...)
```

A `384`-sized matmul dim against two dynamic proxy dims — plausibly the VAE's bottleneck
self-attention block (the same one the *first* session's `_decompose_attention_for_export()`
monkeypatch was written to route around for the native-ONNX-Attention-op dead end; that fix
handles the *op type*, not necessarily this). Bisected with three builds, all at the 480x832
profile, `--precision fp16`:

| `frames` | resolution | export | TensorRT build |
|---|---|---|---|
| 81 | 480x832 | OK | **fails**, `node_matmul_20` |
| 21 | 480x832 | OK | **fails**, `node_matmul_4` (same signature, different node index) |
| 1  | 480x832 | OK | **succeeds** |

**Correction to an earlier over-read of this table:** initially called this "conclusive" —
frame-count-dependent, not resolution-dependent, and not graph-size-dependent since 21 and 81 fail
identically. That reasoning is incomplete: it never included last night's actual known-good
config from `runpod_session_2026-08-06/i2v_smoke_test.py` — `LATENT_FRAMES=3` at **256x256**,
which back-solves to **9 raw frames** (`(9-1)//4+1=3`), not tested at all in this session's matrix.
So the real boundary is somewhere between "9 frames @ 256x256 works" and "21 frames @ 480x832
fails" — and since *both* frame count *and* resolution differ between those two points, whether
the actual cause is frame-count, resolution, or their interaction (e.g. total unrolled-loop-node
count, which scales with both) is still genuinely open. Next step: test 9 frames @ 480x832 and/or
21 frames @ 256x256 to separate the two variables properly, then narrow the frame-count boundary
itself (e.g. 13) if frame-count is confirmed as a real factor.

**Real second bug found and fixed while investigating this**: the `frames=1` test build above
landed on the exact same cache digest (`ad1b3962f07fa491.engine`) as last night's real working
`frames=9`/256x256 engine — and silently overwrote it. Root cause: `runtime/cache.py`'s
`CacheKey.optimization_profile` is a profile *name* string (e.g. `"480x832"`), not the exporter's
actual traced shape — nothing in the key distinguished `VAEEncoderExporter(frames=1)` from
`(frames=9)` or `(frames=81)` when both happened to build under the same profile name. Fixed by
adding `CacheKey.input_shape_digest` (`export/base.py`'s new `ModelExporter.shape_digest()`,
hashing `example_inputs()`'s tensor shapes) and wiring it into both real `CacheKey` construction
sites (`cli/commands/build.py`, `export/pipeline.py` — the latter is the path
`export/pipeline.py`'s own docstring says the ComfyUI "TensorRT Engine Builder" node uses, though
note `cli/commands/build.py`'s `run_engine` does *not* actually call `run_export_pipeline` despite
duplicating its logic — a separate, pre-existing divergence not fixed here). Regression test:
`tests/test_cache.py::test_different_input_shape_is_a_different_cache_entry`. **This means every
engine cached before this fix has an ambiguous shape provenance** — a cache dir built before today
should be treated as untrustworthy for anything beyond single-shape-per-profile use until rebuilt
under the corrected key.

**Status: still an open blocker, not yet root-caused or fixed** — the vae_encoder build failure
itself remains unresolved; only the cache-collision bug it exposed has been fixed so far.

**Follow-up bisection, isolating frame-count from resolution** (now safe from the cache-collision
bug above): built `frames=9 @ 480x832` (same frame count as last night's known-good, larger
resolution) and `frames=21 @ 256x256` (same resolution as last night's known-good, larger frame
count). **Both failed**, same `ForeignNode`/autotuner-can't-find-implementation signature as
before:

| `frames` | resolution | result |
|---|---|---|
| 9  | 256x256 | ✅ works (last night) |
| 9  | **480x832** | ❌ fails |
| **21** | 256x256 | ❌ fails |
| 1  | 480x832 | ✅ works |

Neither frame-count nor resolution alone explains it — `9@256x256` sits just under some real
complexity/size threshold, and pushing *either* variable up alone crosses it. Most likely
explanation: the failing matmul's proxy dimensions are a function of both spatial size and
chunk/frame count (plausibly the VAE's flattened-spatial bottleneck self-attention, whose sequence
length scales with `H*W`, combined with the multi-chunk cross-chunk-merge logic that only exists
at `T>1`), and TensorRT 11.2's autotuner has a real ceiling somewhere in that joint space. This
means the actual target config this project needs (81 frames @ 480x832) is **not** "close, needs
minor tuning" — it's well past a real threshold on both axes independently, matching last night's
smoke test's own docstring instinct to deliberately stay tiny rather than risk exactly this.

**Not attempted / not needed in the end:** polygraphy/verbose per-node TensorRT builder output on
the failing `ForeignNode` to identify the actual op and constraint; whether disabling/adjusting
`_decompose_attention_for_export()`'s specific decomposition changes the failure; whether a
smaller `workspace_limit_mb` or different TensorRT build flags change the autotuner's tactic
search. Pivoted instead — see below.

### Pivot: per-frame `encode_image` design, avoiding the need for a T>1 engine at all

Rather than root-causing the TensorRT build failure, rewrote `_build_image_to_video_conditioning`
to only ever need `T=1` — the one config already proven to build and run. It now calls
`vae_encoder.encode_image()` once per *distinct* pixel content the padded video would contain
(gray, optionally the real first frame, optionally the real last frame — never once per output
latent frame, since e.g. 81 frames only ever has up to 3 distinct contents) and reuses each
result across every latent position sharing that content. `text_encoder`/`vae_decoder`/`dit` are
unaffected; `vae_encoder` goes back to building at its default `frames=1`.

**Correction to the initial framing of this trade-off:** first described it as losing "cross-chunk
causal blending," implying a cost to generation quality/temporal consistency. That conflated two
different things — pushed on this directly and it doesn't hold up. Generation-time temporal
consistency (does frame 40 flow from frame 39) is entirely the DiT's own full self-attention over
the whole latent sequence at every denoising step; it has no dependency on how the *input*
conditioning was VAE-encoded. What the real algorithm's single whole-video `encode_video()` call
actually contributes that this per-frame version doesn't is narrower: a gray frame encoded next to
a real frame picks up a faint trace of that neighbor's pixel content via the VAE's own causal
receptive field, purely a difference in the padding positions' raw *latent values*. The `mask`
channel already tells the DiT exactly which positions are real vs. to-generate regardless, so
there's no dependency on that subtle encoding-time cue for correctness either. Expected to be a
minor, second-order effect — consistent with how the channel-order bug (the actual cause of every
incoherent-output attempt before it was fixed) already dwarfed the gray-fill discrepancy from the
day it was first flagged as "likely smaller-magnitude ... unmeasured."

This also drops the real algorithm's raw-frame-granularity mask asymmetry (first frame:
all-4-channels-known at latent index 0; last frame: only 1-of-4) — that asymmetry was a mechanical
side effect of the real joint chunked encode's raw-frame-granularity mask construction specifically,
and has no principled meaning once each latent frame comes from its own independent encode call.
Mask here is simply all-4-channels-known at a real frame's latent index, all-4-channels-unknown
elsewhere. `tests/test_image_conditioning.py` updated to match (new
`_FakeVAEEncoder.encode_image`, new assertions, plus a test confirming the shared-gray-encode reuse
— only 3 `encode_image` calls total for an 81-frame/21-latent-frame video, not 21).

`build_all.sh`'s vae_encoder step reverted to `frames=1` accordingly — no longer needs the
large-`T` build this section spent so much time on. See docs/roadmap.md.

### First full end-to-end run — real infra bugs found and fixed, but the DiT engine itself is broken

With the pivot above, ran the full pipeline for real on this fresh pod: built all four engines
(text_encoder 20.63GiB, vae_encoder 0.04GiB, vae_decoder 0.21GiB, dit 26.65GiB — dit's size matches
last night's "26.6GiB engine" almost exactly, a good sign the checkpoint/build recipe itself is
consistent), assembled a `model_dir`, and ran `WanEngine.generate()` for real: prompt + first/last
frame images, 480x832, 81 frames, 30 steps, guidance_scale=5.0.

**Two real, unrelated infra bugs found and fixed along the way (both now real gaps in the
standalone API that had simply never been exercised end-to-end before):**

1. **My test script's bug** (not the repo's): `close_green_chair_start.png`/`_end.png` are RGBA
   (4-channel), and my `load_frame()` helper fed all 4 channels straight into a 3-channel VAE
   input — real shape mismatch, fixed by slicing `image[:3]`.
2. **Real repo bug**: `_HFTokenizerAdapter` (`api/wan_engine.py`) used `padding=True`, which pads
   only to the batch's longest sequence — for a short prompt that's far fewer than the 512 tokens
   the DiT engine's `context` input is baked to expect (`DiTExporter.dynamic_axes()` never declares
   a dynamic axis for `context` at all, so it's a hard fixed-length requirement, not a padding
   nicety). Fixed: `WanModelConfig` gained `max_text_tokens: int = 512`, threaded through
   `_HFTokenizerAdapter`/`load_default_tokenizer` (and the ComfyUI `TensorRTWanLoader` node, which
   had the identical latent bug) to `padding="max_length"`.
3. **Real, more serious bug, fixed via a real architecture gap closed**: after the above two fixes,
   generation ran fully through all four engines with zero errors — but `vae_decoder.decode()`
   failed with a genuine ~94GiB (100,756,234,752 byte) execution-context allocation failure at
   *first real inference* (not at `.load()` — TensorRT creates the execution context lazily on
   first `.infer()` call). Root cause: `vae_encoder`/`vae_decoder` were built with a *wide dynamic*
   H/W profile (e.g. `min=32, max=latent_height*2`) even though this project only ever actually
   runs at one resolution per generation call — TensorRT appears to size scratch memory (plausibly
   the VAE's bottleneck self-attention matrix, which scales with H*W) for the profile's worst case,
   not the shape actually used at runtime. This is the exact same root issue as the already-tracked
   Phase 3 item ("Dynamic height/width... likely resolution: switch to Dim.STATIC per resolution
   profile") and the same already-declared-but-dead `ResolutionProfile.dynamic` field — now
   actually wired up: `ModelExporter` gained a `static: bool = False` constructor param: when set,
   `dynamic_axes()` returns `{}` for every subclass (`DiTExporter`/`TextEncoderExporter`/
   `VAEEncoderExporter`/`VAEDecoderExporter`, all updated), which flows through to both
   `torch_export.py`'s `_build_dynamic_shapes` (fully static `torch.export`) and
   `trt_build.py`'s `_build_optimization_profile` (no profile entry needed at all) — the exact
   mechanism that already made `DiTExporter`'s `context` input (never given a dynamic axis) work
   as fully static. Rebuilt `vae_encoder`/`vae_decoder` with `static=true`: **the ~94GiB allocation
   failure disappeared entirely**, confirming the diagnosis. Regression-tested at the unit level
   (`tests/test_export_base.py`).
4. **Real second cache-key gap found while doing this rebuild**: the first static rebuild attempt
   silently served the *old* dynamic-range engine instead of building a new static one — because
   `ModelExporter.shape_digest()` (this session's earlier cache-key fix) only hashed
   `example_inputs()`'s tensor *shapes*, which `static` doesn't change at all (only `dynamic_axes()`
   does). Fixed by folding `dynamic_axes()`'s actual ranges into the digest too, not just shapes.
   Regression test added (`test_shape_digest_differs_between_static_and_dynamic_at_identical_example_shape`).

**With all of the above fixed, the full pipeline ran completely error-free for the first time —
30-step denoising loop, ~5 minutes, saved a valid 81-frame/480x832 mp4.** But the output is
**exactly, uniformly black** (`frame.mean()==0.00, std()==0.00` for every sampled frame; the mp4
itself is 4.6KB, consistent with solid-black H.264 compression). Traced it: `image_latent`/`mask`/
`text_embeds`/initial noise latents are all clean (no NaN, reasonable stats) — but the DiT's very
first `denoise_step` call already returns **100% NaN**, at *every* timestep tested (1, 500, 999,
1000) and *every* guidance_scale tested (1.0, 5.0) — not a single-timestep edge case. Pushed
further: **the DiT engine returns 100% NaN even on literally its own trivial `example_inputs()`
(all-zero `x`/`context`, `timestep=0`)** — the exact shape it was traced/built against, no real
conditioning content involved at all. This conclusively rules out anything from this session's
conditioning-fix work (already independently verified correct via CPU-only tests against fake
stand-ins) — **this is the DiT TensorRT engine build itself producing NaN unconditionally**,
on this fresh pod, despite the engine matching last night's known-working build's file size almost
exactly (26.65GiB here vs. "26.6GiB" last night).

Attempted the eager-vs-TensorRT comparison last night's session used to definitively localize this
kind of bug (`compare_eager_vs_trt.py`, ported to this session's real inputs/channel order as
`compare_eager_vs_trt2.py`) — blocked by an unrelated dtype error in ComfyUI's own
`time_embedding` module (`mat1 and mat2 must have the same dtype, but got Float and Half`),
reproducing regardless of what dtype `timestep` is passed as from the caller side, which points at
something inside `comfy/ldm/wan/model.py`'s `time_embedding`/`comfy.ops` dtype-casting behavior
itself rather than anything in this repo's loader — possibly a ComfyUI version difference between
this fresh pod and last night's (this pod's ComfyUI install was never set up by this repo's own
tooling; it's whatever was already on the persistent `/workspace` volume, see this session's
"fresh pod setup" section above). Not yet resolved — an eager-side comparison would be the fastest
way to confirm whether the checkpoint/environment itself is fine (pointing squarely at the
TensorRT build step) or whether eager is *also* broken in this environment (pointing upstream of
TensorRT entirely).

Re-exported `dit` to ONNX (deleted as a "stale intermediate" earlier this session, before this bug
was found — real lesson: don't clean up `.onnx` intermediates until an engine has actually been
*validated*, not just "successfully built") to check via `onnxruntime-gpu` (already installed)
whether the NaN is present in the ONNX graph itself. Blocked on an unrelated environment gap:
`onnxruntime-gpu`'s installed version needs CUDA 13 (`libcublasLt.so.13` missing; this pod has CUDA
12.8), and CPU-provider fallback got OOM-killed trying to run a 14B-param forward pass on CPU RAM.
Abandoned that path — went straight to the real question instead.

### Root cause found: `comfy/ldm/flux/math.py`'s `rope()` mixes float32/float64 in one Einsum

```python
scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
omega = 1.0 / (theta**scale)  # stays float64
out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)  # float32 x float64
```

Real bug, not a numerical-instability edge case: eager PyTorch tolerates this via implicit type
promotion (computes in float64, `rope()`'s own final `.to(dtype=torch.float32, ...)` discards the
extra precision anyway) — but `torch.export`/ONNX freezes the einsum with two genuinely different
baked-in input dtypes. `onnxruntime` correctly refuses to load the resulting graph at all
(`Type parameter (T) of Optype (Einsum) bound to different types (tensor(float) and
tensor(double))` — this is what a working onnxruntime-gpu install would have caught immediately,
had the CUDA-version mismatch above not blocked it). TensorRT's parser accepts it anyway, and the
built engine then returns NaN unconditionally.

Traced the call path to confirm this fires on every single DiT forward pass, not just a rare
branch: `EmbedND.forward()` (`comfy/ldm/flux/layers.py`) calls `rope(ids[..., i], ...)`
unconditionally for every axis, and `self.rope_embedder(img_ids)` (`comfy/ldm/wan/model.py`'s
`rope_encode`) calls `EmbedND.forward()` unconditionally too — the *other* call site of `rope()` at
`model.py:667` (inside `if source_id:`) is dead code at this project's always-`source_id=0` call
pattern and isn't the actual culprit, a plausible-looking red herring initially considered.

**Fixed**: `wan_comfyui_loader.py` gained `_rope_fp32`, a pure-PyTorch clone of `rope()` with
`scale`/`omega` computed in `float32` throughout instead of `float64`. Wired into `load_dit()` via
`comfy.ldm.flux.layers.rope = _rope_fp32` — critically, patching *that* module's binding, not
`comfy.ldm.wan.model.rope`: `EmbedND.forward()` is defined in `flux/layers.py`, which did
`from .math import ... rope` at its own top level, so the name `rope` resolves against
`flux.layers`'s own namespace at call time, not `wan.model`'s (same principle as the existing
`apply_rope1` monkeypatch, but a different module needed patching here).

**Open question, not resolved:** why didn't last night's (2026-08-06) smaller 256x256/9-frame
build hit this same bug? It's structural — every DiT forward pass goes through this exact call
path regardless of shape. Leading theory (unconfirmed, that pod's gone): last night's session may
have included a live edit made directly on that pod's checkout via `ssh` that was never synced back
to the local Mac repo, and so had no way to carry over to this fresh pod. Given this, adopted a hard
rule going forward (see docs/runpod_setup.md's new opening section): always edit the local repo and
`rsync` up, never edit a pod's checkout directly — a fix that only ever exists live on an ephemeral
pod is a fix that will silently vanish.

**Status at time of writing:** DiT re-export + rebuild with the fix in progress. Not yet confirmed
whether this actually resolves the NaN end-to-end — check this file's next session-log entry (or
git history past this point) before trusting this fix without re-verifying.

### RoPE fix confirmed real but insufficient — two more dtype bugs found via a working eager comparison

Rebuilt `dit` with the `_rope_fp32` fix (28.62GiB, matching the pre-fix size — expected, it's a
tiny math change, not a shape change). **Re-tested with the same trivial all-zero
`example_inputs()` check: still 100% NaN, on every timestep.** So the RoPE fix was real (confirmed:
`onnxruntime`'s graph-load-time type check, which previously failed outright on the Einsum
dtype mismatch, now passes silently) but not sufficient on its own — a second, independent bug was
still present.

Went back to the eager-vs-TensorRT comparison (`compare_eager_vs_trt.py`'s method from the first
session), rewritten against this session's real inputs/channel order as `compare_eager_vs_trt2.py`.
It kept failing with an unrelated-looking error (`mat1 and mat2 must have the same dtype, but got
Float and Half`) inside `comfy/ldm/wan/model.py`'s `time_embedding`/`text_embedding` — looked like
an environment/ComfyUI-version quirk at first. Instrumented the actual tensors directly (added
`.dtype` to every `stats()` print, monkeypatched `sinusoidal_embedding_1d` to print its own
input/output dtypes) instead of continuing to guess, and found two more real, independent bugs —
neither related to RoPE, both pre-existing and simply never exercised before this session's first
real `WanEngine.generate()` runs:

**Bug 1 — `WanEngine._initial_latents()` (`api/wan_engine.py`) called `torch.randn(shape,
generator=generator, device=self.device)` with no `dtype=`, silently defaulting to `float32`**
while every other conditioning tensor in the pipeline is `float16`. Fixed: added
`dtype=torch.float16` explicitly.

**Bug 2 — `_build_image_to_video_conditioning`'s `mask` tensor (`api/wan_engine.py`, this
session's own new code) was built with `dtype=reference.dtype`**, where `reference` is the
caller's *raw pixel* tensor (e.g. `load_frame()`-style helpers commonly do `.float()` and never
cast back down) — no reason to match this project's internal fp16 convention, and the tensor it's
concatenated alongside (`image_latent`, the VAE engine's actual fixed-fp16 output) is what it
should have matched instead. Fixed: `dtype=image_latent.dtype`.

Neither of these two bugs *errors* — `torch.cat` silently promotes the whole concatenated tensor
to the wider dtype rather than raising, so `x` fed into the DiT ends up `float32` with no signal
anything is wrong until something downstream (like eager's strict `F.linear` dtype check) actually
notices. `TensorRTEngineWrapper`'s existing per-input dtype cast (`engine/base.py`, from the first
session's real bug fixes) safely downcasts whatever arrives to the engine's declared dtype before
use, which is *why* neither bug ever raised an error on the TensorRT path either — it just quietly
computed on precision-truncated-then-reinflated data. **Neither of these fully explains the TRT
engine's NaN on its own** (the wrapper's cast is a real numeric conversion, not last night's
byte-reinterpretation bug), but each one had to be fixed before the eager comparison could even
run far enough to find the next thing.

**Bug 3 — the real one, found once the eager comparison finally got past `time_embedding`:**
`context` (text_embeds) is `float32`. Traced directly to the engine binding itself:
`text_embeds dtype=DataType.FLOAT` on the *built engine*, despite `--precision fp16` and every
*input* passing `_validate_precision`'s check. Root cause: `_validate_precision`
(`export/trt_build.py`) only ever checked `network.get_input(i)`, **never outputs** — the same
blind spot as the earlier fp8-mislabeling bug, but biting on a different tensor. `T5`'s real
implementation apparently produces an fp32 final hidden state internally (plausibly a
stability-motivated fp32 LayerNorm — the same category of thing `patch_embedding` needed explicit
handling for on the DiT side), and `_TextEncoderWrapper.forward()` (`wan_comfyui_loader.py`) just
returned it directly with no cast. **Fixed two ways:**
1. `_TextEncoderWrapper.forward()` now explicitly casts its return value to
   `next(self.transformer.parameters()).dtype` before returning.
2. `_validate_precision` now checks `network.get_output(i)` too, not just inputs — so this class
   of bug can't slip through silently again for *any* component, not just this one instance of it.

Rebuilding `text_encoder` with both fixes now — not yet confirmed whether `context` actually comes
out fp16 end-to-end, or whether fixing this (plus the RoPE fix) together finally resolves the DiT's
NaN. Check the next entry in this file before trusting this without re-verifying.

### Decisive result: eager is clean, TensorRT is still NaN — bug isolated to export/build, not conditioning

Rebuilt `text_encoder` with both fixes; `text_embeds` binding now genuinely `DataType.HALF`.
Reran `compare_eager_vs_trt2.py` with all four fixes in place (RoPE fp32, `_initial_latents`
dtype, `mask` dtype, `_TextEncoderWrapper` output cast) on byte-identical `x`/`context`/`timestep`:

```
[x] dtype=torch.float16 nan_frac=0.0000
[context] dtype=torch.float16 nan_frac=0.0000
--- TensorRT engine ---
[trt_noise_pred] dtype=torch.float16 min=nan max=nan mean=nan std=nan nan_frac=1.0000
--- Eager PyTorch (real checkpoint) ---
[eager_noise_pred] dtype=torch.float16 min=-4.0508 max=4.4180 mean=0.0342 std=0.8789 nan_frac=0.0000
eager NaN: False, trt NaN: True
```

**Eager is completely clean — zero NaN, sane stats — on the exact same inputs the TensorRT engine
returns 100% NaN for.** This is the single most useful fact found all session: it conclusively
rules out every conditioning-construction bug fixed above (all four were real and worth fixing,
confirmed by eager now working where it previously errored outright) as *the* cause of the DiT's
NaN. The remaining bug is squarely in the export/TensorRT-build pipeline itself — something that
diverges between what `torch.export`/ONNX/TensorRT actually execute and what real eager PyTorch
computes, despite the ONNX graph now passing every dtype-consistency check found so far (RoPE's
Einsum, the text encoder's output binding).

**Not yet investigated further, and would need real work to localize:** most likely path forward
is a layer-by-layer/block-by-block activation comparison (dump intermediate tensors from both the
eager forward pass and a partial/hooked TensorRT run, bisect which DiT block first diverges) rather
than continuing to guess at the next single dtype mismatch — this could be a numerical-instability
issue (e.g. fp16 softmax overflow in `_decomposed_sdpa`'s attention, or the "Profile kMIN/kMAX not
self-consistent" reshape warnings seen in every DiT build this session turning out to be more than
cosmetic) rather than another clean-cut dtype bug like the previous four. Stopped here to report
back given the time already invested (this session has run ~6-7 hours) rather than open-ended
continued debugging — see docs/roadmap.md.

### Continued: activation-dump bisection, then the real likely root cause

Picked back up with a proper layer-by-layer bisection instead of more guessing. Added
`TRTWAN_BUILDER_OPT_LEVEL` (`export/trt_build.py`, env-var override of
`config.builder_optimization_level`, default explicitly `5` not TensorRT's own implicit `3`) —
build time is almost entirely CPU-orchestrated GPU tactic search, no way to make it GPU-only, but
a lower level searches far fewer candidates: cut this session's remaining DiT rebuilds from ~20min
to ~11min. Useful for iteration, never for a real deployment build.

**Method:** loaded `dit_high_noise.onnx` with `onnx.shape_inference.infer_shapes`, picked 12
tensor names evenly spaced across the graph's 4179 nodes (auto-skipping fp32-by-design islands —
`patch_embedding`'s own conv output, RMSNorm-internal fp32 stability computations — by walking
forward to the next genuinely-fp16 tensor, since `_validate_precision`'s output check, also added
this session, correctly rejects picking those directly), added them as extra ONNX graph outputs
(referencing the same external weights file, no data duplication), rebuilt once, ran on real
inputs. Result, in graph order:

```
idx= 140 (~3%)  clean
idx= 208 (~5%)  clean   -- main image-token stream (32760 tokens)
idx= 626 (~15%) clean   -- text/cross-attn branch (512 tokens)
idx=1044 (~25%) 100% NaN -- main image-token stream -- FIRST DIVERGENCE
idx=1462+       100% NaN on every main-stream tap through to noise_pred
```

Cross-referenced node indices against the raw ONNX op list (op_type + inputs, no rebuild needed)
to make sense of this. **Real finding, independent of the bisection:** `graph.node[611]` and
`[636]` (block 4's self-attention and cross-attention) are ONNX opset 23's **native `Attention`**
op — not the decomposed matmul+softmax+matmul form `_decomposed_sdpa`/
`_decompose_attention_for_export()` exist specifically to force. Checked why:
`_decompose_attention_for_export()`'s own docstring said, explicitly and deliberately: *"Not
applied in `load_dit` itself: the DiT's own attention already finds a dedicated fused kernel with
no such error, so it doesn't need this."* — a real, documented decision from the first session,
not an oversight. But that observation was made against last night's tiny smoke-test scale (~768
attention tokens, `LATENT_FRAMES=3` @ 256x256). Today's real target scale is ~32,760 tokens — over
40x larger. "Found a kernel, no build error" at one scale was never proof of "numerically correct"
at another — this is the exact same shape-dependent-TensorRT-correctness-gap pattern already
independently confirmed for `vae_encoder` this session (builds/works fine small, breaks large).
And the first NaN tap (idx=1044) falls right around block 5's self-attention
(`blocks.5.self_attn`, confirmed by reading the raw op list: RMSNormalization ->
... -> `Attention` at node ~711) — consistent with the native attention op being where this
starts.

**Applied the same fix DiT never got:** `load_dit()` now calls `_decompose_attention_for_export()`
too, matching `load_text_encoder`/`_load_wan_vae`. Updated that function's docstring to correct
the now-superseded claim rather than silently deleting it. Rebuilding+retesting now — not yet
confirmed. This is a well-evidenced hypothesis (documented prior decision, contradicted by a
proven pattern of the same failure mode elsewhere, landing exactly where the bisection pointed),
not a guess, but still unconfirmed until the rebuild actually runs clean. Check the next entry (or
git log past this point) before trusting this fix without re-verifying.

Also ruled out, for completeness: `comfy.ops.RMSNorm`'s `forward_comfy_cast_weights` dynamic-cast
path (a large, stateful, `torch.export`-hostile-looking code path involving live memory-management
state) — checked directly on the loaded model (`self_attn.norm_q`): `comfy_cast_weights=False`,
`weight_function=[]`, `bias_function=[]`, so the plain native `torch.nn.RMSNorm.forward()` path is
what actually runs, not that machinery. Not the cause.

## Update: the decompose-attention-for-DiT fix was rebuilt and tested — it did NOT fix the NaN

Picked back up on the same pod. `load_dit()`'s new `_decompose_attention_for_export()` call was
re-exported to ONNX (`reexport_dit2.log`, 00:29) and rebuilt into a fresh-digest engine
(`rebuild_dit.log`, 01:29–02:00 → `5feea80f12f67933.engine`; same `Profile kMAX/kMIN not
self-consistent` reshape warning as every prior DiT build, already-tracked, not new).

**Retested against the same trivial all-zero `example_inputs()` check used throughout this
investigation (`trivial4.log`, 06:18): still 100% NaN.** Same failure signature as before the fix —
first divergence at the identical debug tap (`idx=1044`, tensor `tmp_0_53`), identical shape.
**This rules out the "native ONNX `Attention` op is fine at toy scale but silently wrong at real
scale" hypothesis** — decomposing it changed nothing about where or whether the NaN appears, so
either the decomposition isn't actually landing in the exported graph, or attention (native or
decomposed) was never the real cause and the bisection's landing spot near block 5's self-attention
was coincidental/downstream rather than causal.

**Not yet checked: whether the monkeypatch is actually taking effect in the graph.** `inspect_range.py`
(reads `dit_high_noise.onnx`'s node list for indices 600–1050, CPU-only, `load_external_data=False`
so it doesn't need the 28GB weights file) was written specifically to check whether the ops around
the previously-identified `Attention` nodes (~611/636) are still the native op or have actually
become decomposed matmul/softmax/matmul — but its output was only ever printed interactively, never
saved to a log file, so this hasn't actually been confirmed either way. **This is the next concrete
step**, before spending more GPU time on further rebuild-and-retest cycles: if the graph still shows
a native `Attention` op there, the monkeypatch itself has a bug (wrong function reference, patched
too late relative to trace time, wrong module namespace — the same class of gotcha the RoPE
monkeypatch already hit once); if it's genuinely decomposed and still NaNs, attention was never the
cause and the bisection needs to be redone against *this* engine's own debug taps (the debug-output
`.onnx` variant, `add_debug_outputs.py`/`debug_output_names.txt`, was built against the
pre-decompose `dit_high_noise.onnx` — needs regenerating against the current graph before trusting
another idx=1044-style localization).

**Also still blocked:** the independent `onnxruntime` cross-check path. `onnx_check3.log`: this
pod's installed `onnxruntime-gpu` refuses to load an opset-23 graph at all ("Opset 23 is under
development... Current official support for domain ai.onnx is till opset 21") — not a real
validation, a version-support gate. CPU-provider fallback still OOMs on a 14B-param forward pass
(pre-existing gap, see the first attempt at this earlier in this doc). Without this, the
eager-vs-TensorRT comparison remains the only working validation method — no way to check "is the
ONNX graph itself already wrong" independent of the full TensorRT build step.

**Status:** open blocker, not root-caused. DiT TensorRT engine still returns 100% NaN on every
input tested, including trivial all-zero ones. Next session should start with the unlogged
`inspect_range.py` check above before attempting another rebuild.

## 2026-08-07 (cont.): fresh pod, confirms the runpod_setup.md golden rule the hard way

Reconnected to the same host:port from `docs/runpod_setup.md`, but it was a genuinely fresh
container — new PIDs, GPU idle, `/workspace/runpod-slim/TensorRT-Wan` gone entirely, engine cache
empty, `tensorrt` not importable. **`inspect_range.py` (and `add_debug_outputs.py`/
`debug_output_names.txt`) were lost** — they only ever existed as live, unsynced files on the prior
pod, exactly the failure mode `runpod_setup.md` warns about. Recreated `inspect_range.py` from the
prior session's description (CPU-only, `onnx.load(..., load_external_data=False)`, prints
`op_type`/inputs/outputs for a node-index range) and put it under `scripts/` in the repo instead of
a `runpod_session_*` dir — the setup rsync's own `--exclude='runpod_session_*'` would have dropped
it again otherwise. Committed to the local repo this time so it survives future pod churn.

Also: this fresh pod had no `rsync` binary at all (`apt-get install -y rsync` after an `apt-get
update` fixed it — the base image apparently doesn't ship it). If a from-scratch pod's rsync sync
step fails with "command not found" / connection-closed, check the remote side has rsync before
assuming a local/network problem.

Redid one-shot setup (`pip install -e ".[tensorrt]" transformers pytest` — transformers was
already present via the base image, everything else fresh: TensorRT 11.2.1.2, onnxruntime-gpu
1.28.0, onnx 1.22.0) and re-ran `export onnx --component dit` (with `load_dit()`'s
`_decompose_attention_for_export()` call, already in the synced repo from last session) to get a
fresh `dit_high_noise.onnx` to inspect.

**`inspect_range.py` result (`trtwan_engines/inspect_range_600_660.log`): decomposition is
genuinely landing in the exported graph.** Nodes 604–625 (block 4's cross-attention) show
`MatMul -> Add(bias) -> RMSNormalization -> ... -> MatMul(qk) -> Mul(scale) -> Softmax ->
MatMul(v) -> Transpose -> Reshape -> MatMul(o)` — the real decomposed matmul/softmax/matmul form,
no native ONNX `Attention` op anywhere in the 600–660 range (block 4's self-attn just before it,
and block 5's self-attn just after, at node 657+, are the same decomposed shape). **This rules out
the monkeypatch-bug hypothesis entirely** (wrong function reference / wrong namespace / patched too
late — none of that is happening; the patch works). Combined with last session's confirmed
rebuild-and-retest (still 100% NaN, same tap `idx=1044`/`tmp_0_53`), this also finalizes ruling out
"native ONNX `Attention` op numerically wrong at real scale" as the cause — decomposing it changed
nothing. **Attention (native or decomposed) was never the root cause.**

Note: total node count dropped slightly (4059 now vs. 4179 in the pre-decompose graph from the
prior session) — decomposition usually adds nodes, not removes them; not investigated further,
likely just an onnx/onnxscript/opset version delta between the two export runs (fresh pip install
today vs. whatever was pinned before) rather than anything meaningful, but flagging in case it
matters later.

**Next step, per the plan going in:** redo the activation bisection against *this* graph — the old
`debug_output_names.txt`/`add_debug_outputs.py` were built against the pre-decompose
`dit_high_noise.onnx` and are now doubly stale (wrong graph *and* lost with the old pod, per the
golden-rule note above). Need fresh evenly-spaced tensor taps on the current graph, one rebuild,
one trivial-input test, same method as the original bisection.

Recreated both scripts (lost with the old pod too) as permanent, committed files this time:
`scripts/add_debug_outputs.py` (CPU-only, evenly-spaces N taps across the graph's nodes, walks
forward past any non-fp16 tensor so `_validate_precision` never rejects a debug output — same
fp32-island-skipping logic as before) and `scripts/bisect_debug_taps.py` (reuses
`TensorRTEngineWrapper` as-is, since it already handles arbitrary named outputs generically via
`engine.num_io_tensors` — no engine-wrapper changes needed). Built the debug-tap engine into a
**separate cache dir** (`trtwan_engines_debug/`, not the real `trtwan_engines/`) since `CacheKey`
doesn't hash ONNX file content/outputs at all, only shape/precision/version metadata — a
debug-tap onnx and the real onnx produce the *same* cache digest (confirmed: both landed on
`5feea80f12f67933`), so building into the real cache dir would've been indistinguishable from (and
could silently overwrite) the production engine.

**Result (12 taps, `trtwan_engines/bisect_debug_taps.log`): NaN starts far earlier than previously
believed.** `idx=13` (`pad`, right after patch_embedding) is clean; `idx=338` (`tmp_0_11`, ~8% into
the graph) is already 100% NaN. The prior session's coarse 12-tap pass (bigger gaps, pre-decompose
graph) had reported first NaN around `idx=1044`/~25% — that was never a precise localization, just
where the *next available tap* happened to land; the true divergence is now known to be
considerably earlier, well within the first block or two, not block 5. `noise_pred` itself is
still 100% NaN, consistent throughout.

Also notable: `idx=1690`/`3042`/`3718` came back **clean** despite sitting topologically after
several already-NaN taps. Not a contradiction — `3042`/`3718` have shape `(..., 512, ...)`
(text-token count), so they're almost certainly on the cross-attention K/V-from-`context` branch,
which never touches the corrupted image-token stream. `1690` is shape `(1, 32760, 5120)` though —
genuinely image-token-shaped — so it's more likely a reused RoPE frequency/table tensor
(input-independent, recomputed/reshaped at multiple call sites by `torch.export`) than the actual
accumulating residual; IEEE-754 NaN propagation through `Add`/`LayerNorm` makes it very unlikely a
tap directly on the real residual chain would read clean after an earlier tap on the same chain
read NaN. Not confirmed which of these two explanations is right — flagged as a caveat on any tap
whose name doesn't obviously read as "the residual stream" (`add_*`, `addcmul_*`, `linear_*`
following one), since it may be measuring an irrelevant parallel branch rather than the thing that
actually matters.

**Immediate next step:** narrow the bisection into `[13, 338]` (currently the only bracketing
clean/NaN pair) to find the actual first-divergence node, then `inspect_range.py` around it to read
the real op. This is now looking like it could land inside the *first* transformer block, or even
inside patch_embedding/positional-encoding setup — worth specifically checking whether it's
upstream of every block (a single shared bug) rather than a per-block accumulation.

### Narrowed bisection: first NaN is inside **block 0's own self-attention**, not block 5

Re-ran `add_debug_outputs.py` with 16 finer taps confined to `[13, 338]` (now takes an optional
node-range so this doesn't need a new script), rebuilt (same `--force`, same debug cache dir — the
CacheKey digest collides with the wider-tap build since it doesn't hash ONNX content, so this
correctly overwrote the now-unneeded first debug engine), reran `bisect_debug_taps.py`
(`bisect_debug_taps2.log`). **First NaN is `idx=213`, `tmp_0_4`** — clean at `idx=195`
(`type_as_1`). Read the raw ops for `[180:320]` (`inspect_range_180_320.log`) to identify exactly
what these are: **this is entirely inside block 0's self-attention** —
`node_matmul`(204)→`Mul(scale)`(205)→`Softmax`(206)→`node_matmul_1`(207) is the decomposed
attention itself (clean going in — Q/K/V and the QK^T/softmax/weighted-sum all check out), then
`linear_8`(212, the `o_proj` bias-add) feeds into `n3_4: Mul(linear_8, getitem_2) -> tmp_0_4`(213)
— **the self-attention output times its modulation gate is where NaN first appears**, immediately
feeding `n4_4: Add(transpose, tmp_0_4) -> addcmul_3`(214), block 0's post-self-attn residual. So:
divergence is not "somewhere around block 5" as the old coarse pass suggested — it's in **block
0**, the very first block, at the self-attention gate multiply. Every block downstream inherits
NaN via the residual stream from here on (consistent with all the NaN taps found afterward, and
with `noise_pred` itself always being 100% NaN).

**New leading hypothesis, not yet tested: TensorRT is still using its own fused attention kernel
even in the "decomposed" graph, so decomposing never actually bypassed the suspect code path.**
Rereading `_decomposed_sdpa`'s own docstring (`examples/loaders/wan_comfyui_loader.py`) — written
last session, easy to have missed the implication at the time — it says outright: *"the resulting
ONNX graph is plain MatMul/Softmax nodes, which TensorRT's older, doc-confirmed MHA fusion pass can
still recognize and fuse."* That fusion is the whole reason the decomposed form was chosen over
leaving the native `Attention` op in (better performance than the native-op import failure). But it
means the "decompose attention to rule out the native op" experiment from earlier this session was
never actually testing "is TensorRT's attention kernel the problem" — TensorRT's builder likely
pattern-matches the exact `MatMul→Mul(scale)→Softmax→MatMul` shape straight back into essentially
the same fused kernel either way. This would fully explain why decomposing "changed nothing."

**Cheap, targeted way to test this without fighting the fusion pass directly:** `trt.BuilderFlag`
includes `STRICT_NANS` — TensorRT's own flag for "propagate NaN through the network per IEEE 754,
even at a performance cost," implying the *default* (unset) does **not** guarantee IEEE-correct NaN
handling in whatever fast-math/fused path it picks. If a fused kernel's default path is producing
NaN as a genuine *computation artifact* (not just propagating already-bad data), enabling this flag
on a rebuild should change the result — either the NaN disappears (confirms a real fast-math bug in
a specific fused kernel, fixable by forcing strict mode or avoiding the fusion) or nothing changes
(rules this out cleanly, no ambiguity). Wrote `scripts/build_strict_nans_test.py` (bypasses
`build_tensorrt_engine()`, reuses its `_build_optimization_profile`/`_validate_precision` helpers
directly, since this is a one-off test rather than a change worth committing to `trt_build.py` until
the result is known) to rebuild the debug-tap engine with this flag set.

**Result: STRICT_NANS made zero difference** (`bisect_strict_nans.log` — byte-for-byte identical
nan_frac at every one of the 16 taps vs. the non-strict build). Cleanly rules out "TensorRT's
default fast-math NaN handling is the bug" — whatever's happening, it's IEEE-consistent NaN
propagation from a genuinely-NaN source, not a strictness/fast-math artifact.

### Decisive test: eager PyTorch at real scale, same trivial input — completely clean

Before spending another rebuild cycle chasing TensorRT-internal theories, ran the actual eager
model (no export, no TensorRT at all) against the identical trivial all-zero inputs at real scale
(`scripts/eager_trivial_check.py`, `eager_trivial_check.log`). Two real gotchas hit writing this,
both worth remembering: **(1)** don't infer dtype from `next(model.parameters()).dtype` — this
model intentionally mixes precision (`patch_embedding` stays fp32), so the first parameter
iterated isn't representative; use `torch.float16` directly, matching
`ModelExporter.dtype`'s own documented reasoning (`export/base.py`). **(2)** `load_dit()`
unconditionally monkeypatches `scaled_dot_product_attention` to the export-only decomposed
reference form, which *materializes the full attention matrix* — at real scale that's `(40 heads,
32760, 32760)` fp16, ~80GiB, instant OOM in eager (94.97GiB card, ~30GiB already resident for
weights). Saved a reference to the original `torch.nn.functional.scaled_dot_product_attention`
before calling `load_dit()` and restored it after, so the eager check runs PyTorch's real
flash/memory-efficient fused kernel instead.

**Result: `noise_pred` came back completely finite — `nan_frac=0.0`, `inf_frac=0.0`,
`min=-0.669`, `max=0.700`.** No NaN anywhere. **This is the most important finding of the
session: the model itself is not broken, and PyTorch's own fused attention kernel handles this
exact scale correctly.** Combined with the decompose-attention and STRICT_NANS results, this
narrows the bug to something specific to **TensorRT's own compiled kernel** for this op at this
scale — not the math, not a degenerate trivial-input edge case, not IEEE-strictness. Matches the
same "works at eager/small scale, breaks specifically in TensorRT at large scale" pattern already
seen with `vae_encoder`, now confirmed for `dit` too, and now with eager ruled all the way in as
the clean reference rather than an untested assumption.

**Next step:** pin down definitively whether the NaN is coming from the attention computation
itself or from the (unrelated, much cheaper) modulation-gate path it gets multiplied against.
`inspect_range_100_183.log` shows node 213's two inputs precisely: `linear_8` (the self-attention
block's own `o_proj` bias-add — the actual attention output) and `getitem_2` (block 0's
self-attention *gate*, sliced from `add_418`, which is purely `blocks.0.modulation` (a learned
per-block parameter) plus `view_14` (the sinusoidal timestep-embedding MLP output) — entirely
independent of attention, cross-attn, or the image/text token streams). Already added both as named
debug outputs (`scripts/add_named_outputs.py`, new script for tapping specific known tensor names
rather than evenly-spaced auto-picking; `dit_high_noise_debug3.onnx`, taps: `linear_8`, `getitem_2`,
`getitem_1`, `getitem`, `pow_4`, `mul_421`, `cos_6`, `sin_6`, `linear`, `linear_1`, `linear_2`,
`view_14`, `add_418` — the full timestep-embedding chain, in case the gate path itself is the
culprit rather than attention). Rebuild + bisect not yet run — see next entry.

**Result (`bisect_debug_taps3.log`): definitive.** `getitem`/`getitem_1`/`getitem_2` and the entire
timestep-embedding chain (`linear`, `linear_1`, `linear_2`, `view_14`, `add_418`) are **all
completely clean** — 0.0 nan_frac, every one. `linear_8` (the self-attention block's own o_proj
output, the actual attention result) is **100% NaN**. This rules out the modulation-gate/timestep
path entirely. **The NaN is produced inside self-attention itself**, somewhere in nodes 196–212
(V-projection → per-head reshape/transpose → `matmul`(204, QK^T) → `Mul`(205, scale) →
`Softmax`(206) → `matmul_1`(207, weighted sum) → transpose/reshape → `o_proj`), given Q/K
(`type_as`/`type_as_1`, nodes 182/195) confirmed clean just before it.

**Tried to narrow further inside the attention op itself — hit a wall that's informative on its
own.** Added `matmul`/`mul_622`/`softmax`/`matmul_1`/`transpose_5`/`_unsafe_view`/`val_657` as named
debug outputs and rebuilt. **The engine failed to load at inference time**: `Requested amount of
GPU memory (88224597504 bytes) could not be allocated` — 88.2GiB, within a few percent of
`40 heads × 32760 × 32760 × 2 bytes (fp16) ≈ 85.9GiB`, i.e. almost exactly the size of the **full,
unfused** self-attention score matrix. This is strong independent confirmation of the fusion
theory from earlier in this session: TensorRT does **not** normally materialize this matrix (it
must be using some fused/tiled kernel to run self-attention within the card's 95GiB at all), but
forcing `matmul`/`softmax` to be actual graph outputs breaks whatever fusion boundary lets it avoid
that — and once broken, the naive unfused form simply doesn't fit in memory. **This closes off
"peek inside the fused kernel via more ONNX-level debug outputs" as a further bisection method on
this hardware** — there's no way to ask TensorRT for an intermediate value inside its own fused
attention kernel without also asking it to stop fusing, which is infeasible at this scale.

### Status at end of session — root cause localized, not yet fixed

**Confirmed, in order of confidence:**
1. The model's own math is correct — eager PyTorch, real 32760-token scale, native fused attention,
   trivial input: completely clean output (`nan_frac=0.0`, finite min/max).
2. TensorRT's built DiT engine returns 100% NaN on the same input, first appearing inside **block
   0's self-attention** (`linear_8`, node 212), not in modulation/gating, not in cross-attention,
   not in the timestep embedding.
3. `_decompose_attention_for_export()` (matmul/softmax/matmul instead of the native ONNX
   `Attention` op) genuinely lands in the exported graph but **does not change TensorRT's runtime
   behavior** — its own docstring already predicted this ("TensorRT's older MHA fusion pass can
   still recognize and fuse" the decomposed form), and the 88GiB OOM when forcing those nodes to be
   real outputs confirms TensorRT is still using some fused kernel either way.
4. `BuilderFlag.STRICT_NANS` (force IEEE-correct NaN propagation) makes no difference — rules out
   fast-math/non-strict NaN handling as the mechanism.

**Working theory:** TensorRT 11.2.1.2's fused/tiled self-attention kernel has a genuine numerical
bug specific to this sequence length (~32760, i.e. block-0-through-noise_pred all inherit NaN via
the residual stream once this first self-attention call corrupts it) on this GPU architecture —
same *class* of scale-dependent TensorRT correctness gap already independently found for
`vae_encoder` this session, now with much stronger localization for `dit`. Not yet proven which
specific fused-kernel implementation TensorRT picked or why it breaks there — that would need
either TensorRT's own verbose/tactic-selection logging (not yet captured) or an
NVIDIA-side bug report.

**Not yet tried (real next steps, none attempted this session):**
- Chunked/tiled self-attention *at the ONNX export level* (genuinely splitting the 32760-token
  sequence into blocks before export, not just a monkeypatch) — would prevent TensorRT's fusion
  pattern-match from applying to the full sequence at once, forcing a different (hopefully correct)
  kernel path. Real implementation work, not a quick script change.
- `trt.IAlgorithmSelector` to enumerate and selectively ban tactics for the self-attention layer,
  if a known-bad fused-MHA tactic can be identified and excluded in favor of a slower-but-correct
  one.
- Try `bf16` instead of `fp16` for just the DiT (not yet attempted — `ModelExporter.dtype` is
  currently hardcoded to `torch.float16` project-wide; would need per-exporter override). Wouldn't
  fix a genuine kernel bug, but would rule in/out an fp16-dynamic-range-overflow variant of the
  same theory cheaply if the bug is actually about numeric range rather than the kernel logic
  itself.
- Verbose/`kVERBOSE` TensorRT builder logging during the self-attention layer's build to see which
  tactic/kernel it actually selects, and whether TensorRT reports anything (a warning, a fallback)
  at that point.

All debug scripts from this session (`scripts/add_debug_outputs.py`, `scripts/add_named_outputs.py`,
`scripts/bisect_debug_taps.py`, `scripts/build_strict_nans_test.py`, `scripts/eager_trivial_check.py`,
`scripts/inspect_range.py`) are committed to the repo this time, not left as pod-only scratch files
— reusable for whichever of the above gets picked up next.

### Fix found and confirmed: query-chunked attention (+ bf16, combined in one test) — 100% NaN → 0% NaN

User asked to pursue the chunked-attention idea and combine it with the bf16 test in one rebuild
rather than two. Implemented both as real code changes, not throwaway scripts:

- **`tensorrt_wan/export/base.py`**: `ModelExporter.dtype` was hardcoded `torch.float16`
  unconditionally. Changed to read the same `TRTWAN_LOADER_DTYPE` env var
  `wan_comfyui_loader.py`'s loaders already use to cast the model itself, so exporter and loader
  can never disagree (previously, requesting a bf16 model load would've silently built fp16
  example inputs against it and hit the exact dtype-mismatch failure mode this property's own
  docstring already warns about).
- **`examples/loaders/wan_comfyui_loader.py`**: `_decomposed_sdpa` now chunks over the query
  dimension (`_ATTENTION_CHUNK_QUERY_LEN`, env-overridable via `TRTWAN_ATTENTION_CHUNK_QUERY_LEN`,
  default 4096) whenever `seq_q` exceeds that threshold and there's no causal mask/attn_mask —
  same exact softmax math per chunk (full K/V per chunk, not an approximation, not online-softmax),
  just computed as several smaller `MatMul→Softmax→MatMul` calls concatenated back together instead
  of one giant one. At the DiT's real self-attention scale (~32,760 tokens) this produces ~9 chunks
  instead of a single (40, 32760, 32760) score tensor. Also applies to cross-attention (query is
  still 32760 image tokens there, even though keys are only 512 text tokens) — harmless extra
  chunking, not shown to be necessary, left in for simplicity rather than adding a second
  key-length-based condition.

Re-exported DiT with `TRTWAN_LOADER_DTYPE=bf16` (now correctly propagates to the exporter too) and
the new unconditional chunking, built with `--precision bf16` into the debug cache dir
(`317b4ee9bc983136.engine`). **Result: `nan_frac=0.0`, `inf_frac=0.0`, `min=-0.652, max=0.668,
mean=0.102, std=0.308`** — closely matching the eager reference (`min=-0.669, max=0.700`) from
earlier this session. Real, working, non-degenerate output. First time this session `noise_pred`
has come back clean.

**Not yet known which of the two changes actually mattered** — bf16 alone (avoiding an
fp16-dynamic-range issue in whatever kernel TensorRT was using) or chunking alone (breaking the
fusion pattern-match that was hitting a scale-dependent bug) could each independently explain this,
or both could be required together. Testing chunking-alone-at-fp16 now (`dit_fp16_chunked.onnx`,
same chunking code, no `TRTWAN_LOADER_DTYPE` override so it defaults to fp16) to isolate the
minimal real fix rather than shipping "switch the whole project to bf16" if it turns out
unnecessary. See the next entry for the result.

**Cleanup:** per-request, deleted the two now-fully-investigated debug engines
(`5feea80f12f67933.engine`, `dit_strict_nans.engine`, ~57GiB total) from `trtwan_engines_debug/`
once their findings were captured in this doc — their logs survive, the 28GiB-per-engine artifacts
don't need to. `trtwan_engines_debug/` is a real disk cost (each DiT engine is ~28.6GiB); worth
periodically clearing entries whose logs are already saved, especially once a real fix lands and
the debug-only builds stop being reference points.

### Isolation: bf16 alone fixes it — the chunking code isn't actually load-bearing for this bug

Re-exported `dit_fp16_chunked.onnx` (default `TRTWAN_LOADER_DTYPE=fp16`, chunking still active
since it's unconditional now) and built it fp16 into the debug cache. **Result: 100% NaN again**
(`fp16_chunked_full_check.log`) — chunking alone, at fp16, does **not** fix it. So the bf16 build's
clean result wasn't from breaking TensorRT's fusion pattern-match after all.

Ran the final isolation: bf16 **without** chunking (`TRTWAN_ATTENTION_CHUNK_QUERY_LEN=999999` so
the chunking branch never triggers, `dit_bf16_nochunk.onnx`, built `--precision bf16`).
**Result: also completely clean** (`bf16_nochunk_full_check.log`) — `nan_frac=0.0`, and the summary
stats (`min=-0.652, max=0.668, mean=0.1016, std=0.3084`) match the bf16+chunked build to full
float precision. Confirmed this is a genuinely fresh, different engine, not a stale cache hit
silently reusing the earlier build — different file size (28,588,586,572 bytes vs.
28,623,403,076 bytes for the chunked version) and a new timestamp, despite landing on the exact
same cache digest as the chunked bf16 build.

**Real conclusion: `bf16` precision alone is the fix. The chunking code, while not wrong, isn't
necessary for this specific bug** — the root cause was fp16 dynamic-range/precision inside
TensorRT's self-attention kernel at ~32,760-token scale, not the single-shot MHA
fusion-pattern-match itself (that theory is now disproven: fp16+chunked still broke it,
bf16+unchunked didn't).

**Real latent bug found along the way, worth its own fix separately:** the bf16-nochunk build
landed on the *same* `EngineCache` digest as the bf16-chunked build and silently overwrote it on
disk. `CacheKey.input_shape_digest` (`ModelExporter.shape_digest()`) hashes `example_inputs()`
shapes + `dynamic_axes()` ranges only — it has no way to detect that the *traced computation graph
itself* changed (chunked attention vs. not) when the declared input/output shapes stay identical.
`--force` correctly bypassed the "use cached engine" check and did rebuild fresh both times (real,
independent builds — confirmed via timestamps/file sizes above), so this wasn't a stale-serve bug
in the sense `troubleshooting.md`'s "Cache seems to be serving a stale engine" section already
describes and rules out — but two *meaningfully different* graphs (different `_decomposed_sdpa`
behavior) silently sharing one cache slot is still a real gap, since anyone rebuilding without
`--force` after an internal code change (no shape change) would get an unexpectedly stale engine
silently. Not fixed this session; worth adding graph/code-version info to the digest input
eventually.

### Fix promoted to production; first-ever full generate() run surfaces a separate, new bug

Per user request: stripped the query-chunking code back out of `_decomposed_sdpa` (confirmed not
necessary — see above), and hardened the `bf16` requirement so it can't be silently regressed:
`load_dit()` (`wan_comfyui_loader.py`) and `DiTExporter.dtype` (`export/exporters/dit.py`) now both
hardcode `torch.bfloat16` unconditionally, ignoring `TRTWAN_LOADER_DTYPE` (logging a `WARNING` if
it's set to anything else) — unlike every other loader in the file, which still follows that env
var normally. `docs/runpod_setup.md`'s build commands updated: `dit` now needs `--precision bf16`
explicitly, not `fp16` like the other three components.

Rebuilt for real this time (not the debug cache): re-exported and rebuilt `dit` at full
`builder_optimization_level=5` (no `TRTWAN_BUILDER_OPT_LEVEL` override — this is the real
artifact, not a debug iteration) into `trtwan_engines/`, confirmed clean (`nan_frac=0.0`) against
the same trivial-input check used throughout this investigation. Also built `text_encoder`,
`vae_encoder`, `vae_decoder` fresh (this pod never had them — the persistent volume's prior builds
were from a now-gone pod), all fp16, all per `runpod_setup.md`'s documented commands. Assembled
`trtwan_model/` — **as symlinks into `trtwan_engines/`, not copies** (caught mid-copy: `shutil.
copyfile` on ~49GiB of engines across a network-backed volume is real wasted time/bandwidth for no
reason, since both dirs live on the same filesystem — `runpod_setup.md`'s own documented snippet
should probably be corrected to symlink by default; not yet done). Cleared out
`trtwan_engines_debug/` and the now-stale debug-tap ONNX variants in `trtwan_engines/` once their
findings were fully captured in this doc (~150GiB combined reclaimed across the session).

**Ran a real I2V `generate()` end to end for the first time ever** (`scripts/run_i2v_generate.py`,
committed) — 81 frames @ 832x480 (the project's actual default target shape), the same
`close_green_chair_start/end.png` test images from the 2026-08-06 session, `bf16` DiT,
`num_inference_steps=20`, `guidance_scale=5.0`. **No crash, no NaN** (output is `uint8`, inherently
can't be NaN, but per-frame std varies realistically across sampled frames rather than being
uniformly zero) — the DiT fix itself is thoroughly confirmed working end-to-end, not just in
isolation. **But the actual video content is pure noise, not a coherent image-to-video result** —
visually confirmed on decoded frames (random per-pixel color, no structure, no resemblance to the
input images).

**This is a distinct, separate, previously-undiagnosed bug** — not a regression from anything this
session touched. `roadmap.md`'s own checklist already listed "Run `WanEngine.generate()` end to end
... compare output against the FP16 PyTorch reference" as *not yet done*, for both T2V and I2V —
this was the first time anyone got far enough (past the DiT NaN) to even attempt it. Skimmed the
likely-relevant code before stopping to report rather than diagnosing further:
`FlowMatchEulerScheduler.step`/`.prepare` (`scheduler/flow_match.py`) and `DiTEngine.denoise_step`'s
CFG formula (`engine/dit_engine.py`) both look structurally correct at a glance (standard
sigma-linspace-with-shift Euler integration, standard `uncond + scale*(cond-uncond)` CFG) — nothing
obviously wrong, so this needs real bisection, not a quick read-through fix. One real gap worth
noting: every DiT correctness check this session (eager-vs-TensorRT, the bf16 isolation, the
production confirmation) used **`timestep=0`** trivial inputs — the *actual* generation loop starts
near `timestep≈1000` (max noise, `shift=5.0`'s sigma≈1.0) and sweeps the full range;
`timestep=0` was never a representative sample of what real denoising actually calls the DiT with,
just a convenient input-independent NaN probe. Not confirmed as the cause, just an untested gap.
Stopped here to report back rather than open a new multi-hour bisection unprompted.

**Follow-up: user identified the likely real cause — missing MoE high/low-noise expert
switching, not a scheduler/CFG bug.** This project only ever built/loads the `high_noise` expert
(`wan2.2_i2v_high_noise_14B_fp16.safetensors`) and runs the *entire* denoising schedule through it
— Wan 2.2's real architecture switches to a separate `low_noise` expert partway through (near-clean
latents are a different regime than near-max-noise ones; the high-noise expert was never trained
for the low-noise end of the schedule). This was already a known, documented gap (`docs/
runpod_setup.md`/build commands comment: "high_noise expert -- Wan2.2 MoE, no expert-switching
implemented"), just not yet connected to an actual observed failure mode until this session's first
real end-to-end run.

Two things done to test this without committing to the full two-pass implementation yet:

1. **Landed real sequential load/unload** (separately good practice, requested independently):
   added `.unload()` to `TextEncoderEngine`/`DiTEngine`/`VAEEncoderEngine`/`VAEDecoderEngine`
   (each delegates to `TensorRTEngineWrapper.unload()`, already existed). `WanEngine.
   from_pretrained()` no longer eagerly loads all four engines (~49GiB simultaneously for no
   reason, since `generate()` only ever uses one at a time); `generate()` now loads/unloads
   text_encoder, then vae_encoder, then dit (for the whole denoising loop), then vae_decoder, each
   around its own stage. Confirmed working via the log timestamps of the next run below (each
   engine loads right before its stage, in order).

2. **Reran with `num_inference_steps=50`** (was 20) to rule out "just needs more steps" cheaply
   before touching the bigger question. **Result: no meaningful change** — `mean=107.72` vs. the
   20-step run's `mean=106.47`, same per-frame std pattern, still pure noise on every sampled frame
   except a faint tan/beige rectangular patch near the last couple of frames (where `last_image`'s
   mask=1 "known" position sits) — consistent with the raw image-conditioning channel visibly
   influencing the decode at that position without the model actually being able to denoise/
   integrate it into a coherent image. This is exactly the signature you'd expect from "the
   conditioning is there, but the model doing the final low-noise refinement isn't" — supports the
   missing-low-noise-expert theory over a scheduler/CFG bug (more steps of the wrong model doesn't
   converge, it just repeats the same wrong regime longer).

**Not yet done:** building the `low_noise` expert as its own DiT engine (same process as
`high_noise`, different checkpoint — presumably `wan2.2_i2v_low_noise_14B_fp16.safetensors` in
`ComfyUI/models/diffusion_models/`, same bf16 requirement should apply, unconfirmed) and wiring an
actual two-pass switch into the denoising loop (`DiTEngine.generate()`/`WanEngine.generate()`) at
whatever timestep/sigma boundary Wan 2.2 actually uses. Real next step, not started — this is a
meaningfully sized addition (another ~28GiB engine build, a real switch-point decision, and testing
that the switch itself doesn't introduce its own bug), not a quick fix.

### Implementing the real two-pass MoE switch

User continued past the report-and-pause point (had ~an hour left). Confirmed
`wan2.2_i2v_low_noise_14B_fp16.safetensors` does exist in `ComfyUI/models/diffusion_models/` on
this pod, alongside `high_noise`.

**Found the real switch rule from ComfyUI's own source, not assumed.** No `boundary`/`moe`/`expert`
string anywhere in `comfy/`'s Python source — the switch isn't in the model code at all, it's a
workflow-graph-level split. `blueprints/Image to Video (Wan 2.2).json` and
`blueprints/Text to Video (Wan 2.2).json` (ComfyUI's own official templates) both embed a subgraph
chaining two `KSamplerAdvanced` nodes: `['enable', ..., 4, 1, 'euler', 'simple', 0, 2, 'enable']`
then `['disable', ..., 4, 1, 'euler', 'simple', 2, 4, 'disable']` — a 4-step schedule split exactly
at step 2, i.e. **50% of total steps**, not a separately-computed sigma value. Both blueprints also
confirm `ModelSamplingSD3` shift=`5.0`, matching this project's `FlowMatchEulerScheduler` default
exactly — a good independent consistency check. Verified this split is step-count-independent in
its actual sigma effect: `FlowMatchEulerScheduler.prepare`'s sigma schedule is
`shift * linspace(1,0,N+1) / (...)`, and `linspace`'s raw value at step-fraction 0.5 is always
exactly 0.5 regardless of `N` — so "switch at 50% of steps" always lands on the same sigma boundary
(~0.833 at shift=5.0) no matter how many inference steps are requested, not just at the blueprint's
own 4-step example.

**Implemented in `tensorrt_wan/api/wan_engine.py`:**
- `WanEngine.__init__`/`from_pretrained` now take `dit_high_noise` (was `dit`) plus optional
  `dit_low_noise`. `from_pretrained` looks for `dit_high_noise.engine` (falling back to the old
  `dit.engine` name for single-expert `model_dir`s) and optional `dit_low_noise.engine`; logs a
  `WARNING` and runs single-expert mode if the low-noise engine is absent, rather than silently
  producing a known-degraded result with no explanation.
- New `WanEngine._denoise()` replaces the old `self.dit.generate(...)` call — owns the scheduler
  loop directly (previously delegated whole-loop control to `DiTEngine.generate()`), switching
  from `dit_high_noise` to `dit_low_noise` at `state.step_index == num_inference_steps // 2`,
  loading the newly-active expert and unloading the previous one right at the switch (same
  load-only-what's-needed pattern as text_encoder/vae_encoder/vae_decoder elsewhere in
  `generate()`). `DiTEngine.denoise_step()` (single-step, engine-agnostic) is reused unchanged;
  `DiTEngine.generate()` itself is now unused by `WanEngine` but left in place for any other
  caller wanting single-expert behavior.

Building `dit_low_noise.engine` now (same export+build process as `high_noise`, bf16, full
`builder_optimization_level=5`) — not yet tested end-to-end. See the next entry for the result.

### MoE switch works mechanically but doesn't fix coherence — VAE round-trip cleared instead

Built `dit_low_noise.engine` (`model_hash=edb89340c8a6fbf1`, confirmed genuinely different from
`dit_high_noise`'s `c21c21efa368d529` — not a duplicate/cache collision), symlinked both
`dit_high_noise.engine`/`dit_low_noise.engine` into `trtwan_model/`, reran the same real I2V
generate() (81 frames @ 832x480, 50 steps). **Log confirms the switch fires at the right point**
(`dit_high_noise` loads, denoises, `dit_low_noise` loads ~5.5min later — the halfway point of a
~11min DiT stage — `vae_decoder` loads last). Output distribution genuinely changed (`mean=145.13`
vs. the single-expert run's `106.47`, lower per-frame std 27-49 vs. 40-70) confirming the low-noise
expert really is running different math, not silently failing over to the same thing.

**Still pure noise, no coherent structure, on every sampled frame.** The two-pass MoE switch is
mechanically correct but **did not fix the underlying bug** — ruling MoE out as the (sole) cause.

**Decisive follow-up: bypassed the DiT/scheduler entirely and tested the VAE round-trip in
isolation** (`scripts/vae_roundtrip_check.py`, new) — `vae_encoder.encode_image()` on a real photo,
immediately `vae_decoder.decode()` the result back (padded to the decoder's required 21-frame
shape by repeating the single encoded frame, not the real algorithm, just enough to isolate the
round-trip), no DiT/text/scheduler involved at all. **Result: clearly recognizable** — the decoded
image is unmistakably the same green chair against the same cream wall/window as the input, some
softness and a faint checkerboard artifact (from the frame-repeat hack, not a real bug) but
genuinely coherent. **This rules the VAE encoder and decoder out entirely** as the source of the
noise problem — both are numerically correct.

**Status: DiT (confirmed correct via eager-vs-TensorRT at `timestep=0`, and now with MoE expert
switching mechanically working) and VAE (confirmed correct via direct round-trip) are both cleared.
The bug is somewhere else in the pipeline** — real remaining candidates, none yet tested:
- **The DiT was only ever numerically validated at `timestep=0`** (a convenient input-independent
  NaN probe, never a representative sample of real generation) — real denoising sweeps
  `timestep≈1000` down to `0`; nobody has checked eager-vs-TensorRT agreement, or even eager
  sanity, at a realistic high-noise timestep with real random latents.
- Text embedding correctness — `google/umt5-xxl` vs. ComfyUI's own SentencePiece tokenizer was
  flagged unconfirmed back in `runpod_setup.md` and never independently checked.
- Something in how the scheduler's `timestep`/`sigma` convention (raw `[0, 1000]` range) matches
  what the DiT actually expects, or a scale mismatch between the initial noise latents and what a
  flow-matching schedule assumes.
- CFG sign/magnitude, though the formula read as standard on inspection earlier.

Session paused here — user had limited remaining time. Next session should start with a real
eager-vs-TensorRT DiT comparison at a realistic high-`timestep` sample (not `timestep=0`), since
that's the largest untested gap in an otherwise now-cleared pipeline.

### BREAKTHROUGH: real ComfyUI pipeline + our TensorRT DiT = coherent output. The bug was never the DiT.

User kept going. After several more failed hyperparameter attempts (aspect-crop, per-expert CFG,
mask-polarity flip, real whole-video VAE encode — see the entries above/below), and after fixing a
severe TensorRT dynamic-profile OOM discovered while building genuinely dynamic-resolution DiT/VAE
engines (see the "dynamic size support" entries), user proposed the actually decisive test: **stop
reimplementing ComfyUI's pipeline and just plug our TensorRT DiT into the real, already-proven
ComfyUI workflow** (`user/default/workflows/wan-slim-example.json`, the `custom_nodes/spnxx`
package) — real CLIP, real VAE, real `WanFMLFPluggable` conditioning, real `two_phase_sampler`,
substituting *only* `diffusion_model` for our TensorRT engine.

**Wrote `scripts/real_pipeline_trt_dit_test.py`** — loads real CLIP (`comfy.sd.load_clip`) and VAE
(`comfy.sd.VAE`) exactly like `CLIPLoader`/`VAELoader`'s own code, loads real model shells via
`comfy.sd.load_diffusion_model()` (to get correctly-configured `model_sampling`/`latent_format`/
`concat_keys`, discarding the real weights), then replaces `.model.diffusion_model` with a
`TRTDiTWrapper(nn.Module)` that calls our `TensorRTEngineWrapper` internally. Calls
`WanFMLFPluggable.execute()` and `two_phase_sampler()` directly (real custom-node functions, not
reimplemented). Deliberately narrowed vs. the full reference: no lightx2v LoRA (our engine has none
applied), no CLIP vision (our exported graph has no `clip_fea` input), 480x832 landscape (not the
portrait native resolution) — one variable at a time, isolating specifically "does our TensorRT DiT
converge under 100% real everything-else."

**Real integration gotchas hit and fixed, each a genuine, generalizable finding:**
1. `custom_nodes/spnxx/__init__.py` pulls in a `video.py` submodule that assumes a running
   `server.PromptServer.instance` at import time — mocked just enough (`routes` attribute
   returning no-op decorators) to import the two node functions actually needed, without starting
   a real server.
2. `WAN22.concat_cond` (`comfy/model_base.py`) introspects
   `diffusion_model.patch_embedding.weight.shape[1]` to compute extra conditioning channel count —
   gave `TRTDiTWrapper` a real (tiny, unused-in-forward) `nn.Conv3d(36, 1, kernel_size=1)` so that
   introspection succeeds.
3. **ComfyUI's real CFG batches cond+uncond into one `batch=2` call by default** — our exported DiT
   graph has `batch` specialized to `1` at `torch.export` time (confirmed, matches
   `DiTExporter.dynamic_axes()`'s own documented finding). `TRTDiTWrapper.forward()` now splits any
   `batch>1` input into `batch=1` calls and re-concatenates the outputs — exactly what this
   project's own `DiTEngine.denoise_step()` already does for its own (separately-invoked) CFG
   passes, just needed here because ComfyUI's *default* convention batches them together instead.
4. Our exported `context` input has no dynamic axis (`max_text_tokens=512` baked in) — ComfyUI's
   real tokenizer doesn't necessarily pad to exactly that length, so `TRTDiTWrapper.forward()`
   pads/truncates defensively.
5. `comfy.sd.VAE.decode()`'s internal tiled-decode path applies an in-place `.add_()/.div_()/
   .clamp_()` to its own freshly-computed output tensor, which failed with "Inplace update to
   inference tensor outside InferenceMode" — this session's whole call stack appears to run under
   a persistent/global inference-mode-like state (cloning just our own input tensor didn't help,
   since the failing tensor is computed fresh internally by VAE decode, not derived from our
   input). Fixed by monkeypatching `vae.process_output` to clone immediately before ComfyUI's own
   in-place chain runs.

**Result: `steps=12` (`high_cfg=1.8, low_cfg=1.1`, real values from the reference workflow),
`sampler=euler, scheduler=sgm_uniform` — completed both sampling phases with zero errors, and the
decoded video is a clean, fully coherent, recognizable green chair matching the source image,
staying coherent across all 81 frames (checked first/middle/last), with camera motion consistent
with the prompt. `scripts/real_pipeline_trt_dit_test.py`'s output:
`frames: shape=(81, 480, 832, 3) mean=92.15, per-frame std=~58-65` (all real image statistics, not
noise-like).**

**Conclusion: the entire "pure noise" investigation across this whole extended session was never a
DiT or TensorRT bug.** Everything confirmed about the DiT this session (NaN fix via `bf16`, MoE
switching, numerical sanity at every tested timestep) was correct and remains correct. The bug was
somewhere in *this project's own* reimplementation of the surrounding pipeline — and since this
test reused ComfyUI's real CLIP, VAE, conditioning construction, CFG, and sampler/scheduler
wholesale (only the DiT was ours), the remaining suspect is narrowed specifically to whatever this
test did NOT reuse from our own code: **`FlowMatchEulerScheduler` (our own shift-based linear sigma
schedule) and/or `WanEngine._denoise()`/`DiTEngine.denoise_step()`'s own CFG loop, versus
ComfyUI's real `model_sampling`-driven `sgm_uniform` schedule and `comfy.samplers.sampling_function`
CFG.** Every one of this session's other fixes (latent normalization, mask polarity, real
empty-string CFG embedding, whole-video VAE encode) were real, correct, worthwhile fixes already
landed in `wan_engine.py`/`dit_engine.py` — they just weren't the actual remaining gap, which lives
in the scheduler/sampling-loop math specifically.

**Immediate next step, not yet done:** compare `FlowMatchEulerScheduler.prepare()`'s sigma schedule
and `.step()`'s update rule directly against ComfyUI's real `model_sampling`/`sgm_uniform`
implementation (`comfy/model_sampling.py`, `comfy/samplers.py`'s scheduler-name-to-sigma-function
dispatch) to find the actual numeric discrepancy, then fix our own scheduler to match. This is a
much smaller, well-scoped, high-confidence fix compared to everything else attempted this
session — the DiT is proven correct, so this is purely a scheduler/sampling-math correctness task
now, not an open-ended bisection.

### User kept going. Tried steps/CFG/aspect-crop tuning — none of it worked; then found the real bug

User had ~an hour more and asked to keep going. Answered a side question first: confirmed mask
polarity is `1=known/already-encoded, 0=needs generation`, directly sourced from `dit_engine.py`'s
own docstring (which traced ComfyUI's real `WanImageToVideo`/`concat_cond`) for the legacy path;
the standalone-API path (`_build_image_to_video_conditioning`) implements the same polarity in code
but doesn't independently re-state the ComfyUI citation for itself specifically.

**Three tuning attempts, in order, none of which fixed anything:**
1. **Aspect-ratio-preserving crop.** The real test images are 720x1088 (portrait); every prior run
   naively `.resize()`'d them straight to 832x480 (landscape), squishing the whole scene. Fixed
   `scripts/run_i2v_generate.py`'s `load_image()` to center-crop to the target aspect ratio before
   resizing (per user: fix this on the pod side, no engine rebuild needed — the built engines
   already support any input that resizes to their fixed 832x480 shape, the bug was purely in
   preprocessing).
2. **Per-expert guidance scale.** User's insight: by the time the low-noise expert takes over,
   coarse structure is already set and remaining steps mostly add detail — flat CFG across both
   phases can over-guide the detail phase. Added a real `guidance_scale_low_noise` parameter to
   `WanEngine.generate()`/`_denoise()` (defaults to `None` = same as `guidance_scale`, backward
   compatible) rather than hardcoding a guess. Tested `guidance_scale=3.0`,
   `guidance_scale_low_noise=1.0`.
3. **Result: still pure noise**, and actually *less* structured than before (per-frame std dropped
   to 16-35 from 27-49; even the faint tan patch at the known-frame position, present in every
   earlier attempt, was no longer visible). None of steps/MoE/CFG/aspect-crop moved the needle at
   all — strong evidence the bug isn't in any of these surface parameters.

**Decisive test: eager PyTorch DiT, zero TensorRT.** Wrote `scripts/eager_dit_full_generate.py` —
reuses the already-trusted TensorRT text_encoder/vae_encoder/vae_decoder (text embedding was never
independently questioned; the VAE round-trip was directly verified), but runs the actual DiT
denoising loop (both experts, real switch, real CFG) via `load_dit()`'s eager PyTorch model, no
export/ONNX/TensorRT involved for the DiT at all. Two real bugs hit writing it, both fixed inline:
**(1)** same OOM-prone decomposed-attention monkeypatch gotcha as `eager_trivial_check.py` earlier
this session — restored native SDPA after `load_dit()`. **(2)** `load_dit()` now hardcodes `bf16`
(this session's earlier fix) but the TensorRT text/VAE engines are still fp16 — had to explicitly
cast their outputs to bf16 before feeding the eager bf16 model, or every Linear call raises "mat1
and mat2 must have the same dtype".

**Result: `nan_frac=0.0` at every one of 20 steps, `pred_std` stays sane throughout (confirms the
MoE switch fires at `t=832`, matching the ~0.833 sigma boundary estimate exactly) — but the decoded
video is still pure noise.** This is the single most important result of the session:
**it completely rules out TensorRT.** The bug is upstream of TensorRT entirely, shared by both the
eager and TensorRT paths — something in the conditioning construction, the scheduler, or the model
call convention itself.

**Root cause found — sourced directly from real ComfyUI code, not guessed.** Grepped
`comfy/model_base.py` and `comfy/samplers.py` for `process_latent_in`/`process_latent_out`
(`comfy/latent_formats.py`'s `Wan21` class: a real per-channel mean/std normalization, 16 published
values, applied via `process_in`/`process_out`) and found **two real call sites this project never
replicated**:
- `WAN21.concat_cond()` (`model_base.py`) — the *exact* function this project's own
  `_concat_image_conditioning` docstring already cites for channel order — applies
  `process_latent_in` to the image-conditioning latent before channel-concatenating it into `x`.
  This project fed the VAE encoder's raw output directly instead.
- `samplers.py`'s `inner_sample()` — `return self.inner_model.process_latent_out(samples...)` —
  applies the inverse to the **final denoised latent** before it's ever handed to the VAE decoder.
  This project fed the DiT's raw output straight to `vae_decoder.decode()` instead.

The initial noise latent does *not* need this (flow matching draws it directly as unit normal,
already in the space `process_in` maps real latents into — confirmed via `inner_sample`'s own
"don't shift the empty latent image" skip for an all-zero starting point, the T2V/I2V-from-scratch
case this project uses).

This fully explains every earlier result: the VAE round-trip test (encode→decode, no DiT) never
needed this and was correctly coherent; the trivial `timestep=0` NaN checks never cared about real
data distribution; every generation attempt fed the DiT out-of-distribution conditioning and then
decoded an out-of-distribution output, regardless of how many steps, which CFG, which expert, or
which crop was used.

**Implemented in `tensorrt_wan/api/wan_engine.py`:** `_WAN21_LATENTS_MEAN`/`_WAN21_LATENTS_STD`
(the real published 16-channel values) plus `_wan_latent_process_in`/`_wan_latent_process_out`
helpers. Applied `process_in` to `image_latent` at the end of
`_build_image_to_video_conditioning`, and `process_out` to `latents` in `generate()` right before
`vae_decoder.decode()`. Also patched `scripts/eager_dit_full_generate.py` the same way to
cheaply confirm the fix in the eager path before re-testing the full TensorRT pipeline (much
faster iteration than a real generate() call). **Not yet confirmed working — testing now.**

**Not yet checked:** the legacy ComfyUI-graph conditioning path (`_concat_image_conditioning` in
`dit_engine.py`, used by the TensorRT ComfyUI custom nodes, not the standalone `WanEngine` API)
likely has the same missing-normalization gap for the image-conditioning latent specifically — but
a real ComfyUI graph's stock `VAEDecode` node may already apply `process_latent_out` on its own
side for the *output* half, so that half might not need the same fix there. Not investigated this
session; only the standalone API path (what's actually been under test) was fixed.

Decided (user): stripped the chunking code back out, hardened `bf16` against silent regression.
See the "Fix promoted to production" entry above for what actually landed and the new bug it
surfaced.

## 2026-08-08 session: Refit-API LoRA support — groundwork findings (not yet implemented)

LoRA currently has **zero effect** on the TensorRT DiT path by design: `TensorRTDiTModule.forward()`
(`comfyui/nodes/dit_loader.py`) calls the compiled TensorRT engine directly and never reads the
wrapped model's `nn.Module` parameters. ComfyUI's LoRA nodes patch `ModelPatcher`'s parameter dict
(`add_patches` → `patch_model()` mutates real weights in place); our engine's weights are baked
into the `.engine` file at build time, so any LoRA patch silently lands on the dummy
`patch_embedding` Conv3d (which exists only so `WAN22.concat_cond` can read its weight shape) and
never touches actual computation. Confirmed empirically (user loaded a real LoRA in the ComfyUI
workflow, no visible effect) before this was traced to root cause.

**The real fix is TensorRT's Refit API**, confirmed available on this pod's TensorRT 11.2.1.2:
`trt.BuilderFlag.REFIT` and `trt.Refitter` both exist (`python3 -c "import tensorrt as trt;
print('REFIT' in dir(trt.BuilderFlag)); print(hasattr(trt, 'Refitter'))"` → `True True`). Current
`tensorrt_wan/export/trt_build.py` does **not** set the REFIT flag — existing DiT engines
(including the pinned known-working ones) are not refittable as built; a refit-capable engine
needs a rebuild with that flag set.

**Real Wan LoRA key formats surveyed** (all from real checkpoints in
`/workspace/runpod-slim/ComfyUI/models/loras/` on the pod) — two distinct naming conventions in
the wild, both keyed on the identical base module path:

- `wan-lightx2-high.safetensors`: `diffusion_model.blocks.{i}.{submodule}.lora_down.weight` /
  `.lora_up.weight`, rank **64**. Also has bias/norm deltas: `.diff_b` (bias delta),
  `.diff` (norm delta, e.g. `norm_k.diff`/`norm_q.diff`), `.diff_m` (per-block modulation delta,
  shape `[1, 6, 5120]`) — these apply directly (add to base), not a low-rank product.
- `wan-svi-2-pro-high.safetensors`: `lora_A.weight`/`lora_B.weight` (down≡A, up≡B, same
  `delta = scale·(up@down)` math), rank **128**, weight-only (no diff_b/diff/diff_m).
- `DR34ML4Y_I2V_14B_HIGH_V2.safetensors`: same `lora_A`/`lora_B` convention, rank **32**.
- `wan-4lex.safetensors`: same `lora_A`/`lora_B` convention, rank **32**.

Example full key set for one block (`wan-lightx2-high`, block 0, `cross_attn` + `ffn`):
```
diffusion_model.blocks.0.cross_attn.{k,o,q,v}.diff_b            [5120]
diffusion_model.blocks.0.cross_attn.{k,o,q,v}.lora_down.weight  [64, 5120]   (q/k/v/o all 5120-dim)
diffusion_model.blocks.0.cross_attn.{k,o,q,v}.lora_up.weight    [5120, 64]
diffusion_model.blocks.0.cross_attn.norm_k.diff                 [5120]
diffusion_model.blocks.0.cross_attn.norm_q.diff                 [5120]
diffusion_model.blocks.0.diff_m                                 [1, 6, 5120]
diffusion_model.blocks.0.ffn.0.diff_b                            [13824]
diffusion_model.blocks.0.ffn.0.lora_down.weight                  [64, 5120]
diffusion_model.blocks.0.ffn.0.lora_up.weight                    [13824, 64]
diffusion_model.blocks.0.ffn.2.diff_b                            [5120]
diffusion_model.blocks.0.ffn.2.lora_down.weight                  [64, 13824]
diffusion_model.blocks.0.ffn.2.lora_up.weight                    [5120, 64]
```
(`ffn.0` = up-projection 5120→13824, `ffn.2` = down-projection 13824→5120 — matches Wan's DiT FFN
dims. `cross_attn.{q,k,v,o}` all operate at the 5120 model dim.) 1500 total keys for
`wan-lightx2-high` (rank-64 + diffs), 800 total keys for the three rank-32/128 `lora_A/B`-only
files (no diffs).

**Still open / next step:** whether our exported DiT ONNX's initializer names match this
`diffusion_model.blocks.{i}.{submodule}.weight` convention (minus the `diffusion_model.` prefix,
presumably, since our export traces the bare `WanModel`, not a full checkpoint) — this determines
whether refit is a straight name-based lookup or needs an explicit mapping table between ONNX
initializer names and LoRA checkpoint key names. Export in progress at time of writing
(`dit_high_noise.onnx`, `--exporter-kwargs '{"in_channels": 36, "text_dim": 4096}'`); check this
doc's next entry or `git log` for the result before assuming either way.

**Design input from user (mid-investigation):** the LoRA UX should work like a normal ComfyUI
multi-LoRA loader node (model in, model out) if at all possible, OR a custom TensorRT-specific
LoRA picker node that performs the refit per selected LoRA — not a requirement to select LoRAs
only at `trtwan build engine` time (which is what Phase 3's roadmap bullet originally assumed
before Refit was investigated).

**Resolved (same day, later session): name-based lookup does NOT work — need a graph-traversal
mapping table.** Inspected `dit_high_noise.onnx`'s 1216 initializers (`load_external_data=False`,
no need to touch the 26GB of weight data). Only biases and norm weights kept clean
`blocks.{i}.{submodule}.{bias,weight}` paths. The actual `q/k/v/o`/`ffn.0`/`ffn.2` **weight**
matrices — the ones LoRA patches — lost their parameter names entirely: torch.export's
decomposition transposes `nn.Linear.weight` before it feeds `MatMul`, which bakes it into a new
constant with a synthetic `val_NNNN` name instead of preserving the original path. Confirmed by
shape: 321 `val_*` initializers are `[5120,5120]` (40 blocks × 8 attn projections, +1), 40 are
`[5120,13824]` (`ffn.0`), 40 are `[13824,5120]` (`ffn.2`) — exact expected counts.

The recovery is a clean one-hop graph walk, not a heuristic: every bias-add node's sibling input is
the `MatMul` whose weight input is the corresponding `val_NNNN` initializer. Verified across
submodule types and block indices (0, 5, 39):

```
blocks.0.self_attn.q.bias  -> val_570  [5120, 5120]
blocks.0.self_attn.k.bias  -> val_592  [5120, 5120]
blocks.0.cross_attn.o.bias -> val_689  [5120, 5120]
blocks.0.ffn.0.bias        -> val_696  [5120, 13824]   (note: [in,out], transposed vs PyTorch's [out,in])
blocks.0.ffn.2.bias        -> val_698  [13824, 5120]
blocks.5.self_attn.q.bias  -> val_1357 [5120, 5120]
blocks.39.ffn.0.bias       -> val_6819 [5120, 13824]
blocks.39.cross_attn.o.bias-> val_6812 [5120, 5120]
```

Algorithm: for each known bias name, find the single `Add` node consuming it, take its other input
(a `MatMul` output), and that `MatMul` node's initializer-typed input is the weight. This is
deterministic and cheap (graph-only, no weight data needed) — compute once per exported ONNX and
cache the `(block_idx, submodule) -> onnx_weight_initializer_name` table as a JSON sidecar.

**Still open:** whether TensorRT's `Refitter.set_named_weights()` addresses weights by this same
ONNX initializer name, or by a layer name TRT assigns during optimization/fusion (which may differ
if the builder fuses the transpose into the MatMul, or fuses MatMul+Add into one node) — need a
REFIT-flagged engine build to check `trt.Refitter.get_all_weights()`'s actual names against this
table before assuming they match 1:1. Existing pinned engines are not refit-capable as built
(REFIT flag not set in `trt_build.py`); this needs its own engine rebuild to test, separate from
tonight's known-working rebuild.

## 2026-08-09 (cont.): Refit-API validated end-to-end (structurally) — plus two build-speed findings

**Resolved: `REFIT_INDIVIDUAL` is the right flag, not plain `REFIT`.** Checked NVIDIA's actual
docs (not memory) before committing to an approach: plain `REFIT` marks *every* weight refittable
and is documented to break more fusions than necessary for no benefit here (LoRA only ever touches
attention/FFN weights, never biases/norms). `REFIT_INDIVIDUAL` + `network.mark_weights_refittable(name)`
per-weight is strictly better — and per NVIDIA's docs, the ONNX parser already propagates ONNX
initializer names into TensorRT's weight identifiers by default, so no separate name-mapping layer
is needed at all. Also confirmed `REFIT_IDENTICAL` is a different, incompatible flag (assumes refit
values equal build-time values — undefined behavior otherwise) that must never be used for real
LoRA deltas. `ENABLE_TACTIC_HEURISTIC` was considered and dropped — doesn't exist in TRT 11.2
(`'ENABLE_TACTIC_HEURISTIC' in dir(trt.BuilderFlag)` → `False`; removed after TRT 8.x).

Implemented in `tensorrt_wan/export/trt_build.py`: `_lora_refittable_weight_names(onnx_path)`
generalizes the bias→Add→MatMul graph walk across every block (self-terminates when a block index
matches nothing, so it doesn't assume 40 blocks), found exactly 400 names (40 blocks × 10
submodules: self_attn/cross_attn {q,k,v,o} + ffn.{0,2}), zero duplicates. `TRTWAN_ENABLE_REFIT=1`
now sets `REFIT_INDIVIDUAL` and marks exactly those 400 weights via `mark_weights_refittable()`.

**Validated on a real REFIT-flagged build of `dit_high_noise`** (isolated cache dir
`trtwan_engines_refit_test/`, never touches the pinned known-working engines — `CacheKey.digest()`
doesn't include the REFIT flag or opt level, so building into the *same* cache dir would have
silently overwritten the production pin under the identical digest; confirmed this risk before
running anything): all 400 `mark_weights_refittable()` calls succeeded, and after building,
`trt.Refitter(engine, logger).get_all_weights()` returned **exactly those same 400 names — perfect
match, zero discrepancies either direction**. This closes out the "still open" question above:
the whole pipeline (recover names from ONNX → mark refittable → build → Refitter sees exactly
those names) works structurally end-to-end. Not yet validated: actually computing a LoRA's delta
and calling `refitter.set_named_weights()` + `refit_cuda_engine()` to confirm the *applied* engine
produces visibly different output — that's the next step, wiring into `comfyui/nodes/dit_loader.py`
(currently a no-op for LoRA, see the entry above) so it can be tested with a real LoRA loaded in a
ComfyUI workflow.

**Build-speed finding #1 (real, applied): skip the `bytes(serialized)` copy.** Added
`time.monotonic()` phase instrumentation to `trt_build.py` per a debugging suggestion. Initially
misread a ~22min gap in the `high_noise` build's log timestamps (between the timing-cache-write log
and `EngineCache.put`'s "Cached engine at" log) as proof `bytes(serialized)` was the cost —
reasonable at the time since no direct measurement existed yet, but **wrong**: once instrumented
directly, the `low_noise` build's `bytes(serialized)` conversion measured **26.7s**, not 22 minutes.
The original 22min gap was much more likely ordinary GPU/CPU contention from a concurrent ComfyUI
workflow test running in that same window — the same contention pattern seen repeatedly all
session whenever ComfyUI ran alongside a build. Correcting the record here rather than leaving the
wrong claim standing. The fix itself is still valid and kept: `IHostMemory` implements the buffer
protocol (confirmed via introspection: `nbytes`, `__buffer__`), and `pathlib.Path.write_bytes()`
already wraps its argument in `memoryview()` internally (confirmed by reading its source) — so
`build_tensorrt_engine()` now returns the raw `IHostMemory` instead of copying it into a `bytes`
object first. Verified byte-identical via a real dummy-engine round-trip (sha256 match) before
applying it to the real path. Real, positive, just a ~27s win on a 28GB engine, not the ~22min one
originally claimed.

**Build-speed finding #2 (real, unresolved): the actual disk write to `/workspace` can be very
slow under contention.** Measured `/workspace` (MooseFS network mount) at **689MB/s** via `dd`
when nothing else was running — 28.6GB would take ~41s at that rate. But mid-`low_noise`-build,
with ComfyUI running again (user had restarted it to test something), the actual engine write was
observed crawling at **~17MB/s** (12.84GB written over ~12.5 minutes before the build was killed) —
40x slower than the clean measurement, and CPU-bound-looking (~97%, one core) rather than
I/O-wait-looking, which doesn't obviously fit a pure network-throughput explanation and wasn't
root-caused before the build was stopped. Timing cache and ONNX export both completed fast and
clean earlier in the same session, so this looks specific to the final large write under
concurrent load, not a general MooseFS problem. Worth instrumenting the write step itself
(not just bytes-conversion) next time this comes up, rather than assuming either "network storage"
or "CPU contention" without measuring.

**Build-speed finding #3 (real, unresolved): the timing cache did NOT meaningfully speed up an
identical rebuild.** Rebuilt `dit_low_noise` from scratch after the killed attempt (same ONNX, same
shapes, same REFIT flags, same opt level, timing cache fully populated from both the `high_noise`
build and the killed `low_noise` attempt's own completed tactic search) expecting a near-total
cache hit. Actual result: tactic search took **1257.1s**, versus the killed attempt's **1267.5s** --
a ~10s difference, not the dramatic cut expected from a supposedly-hot cache. Plausible explanation
(not confirmed): a timing cache only persists the *measured timing* of each candidate tactic, not
the cost of compiling/instantiating those candidates in the first place -- if kernel
JIT/instantiation dominates over the timing-measurement step itself, caching timings wouldn't help
much. Also plausible: `REFIT_INDIVIDUAL` disabling fusion across the 400 marked weights means more,
smaller, individually-instantiated layers than a non-refit build, and that per-layer instantiation
cost may simply not be cache-eligible the way tactic timing is. Neither explanation confirmed --
flagging as open rather than asserting either one.

**End-to-end validation: both experts pass.** `dit_low_noise` REFIT build (`7d16ae577fe5bc92.engine`,
matches the known-working digest) also validated clean: `Refitter.get_all_weights()` on the
deserialized engine returned exactly the same 400 names `_lora_refittable_weight_names()` predicted
from its ONNX -- zero discrepancies, same as `dit_high_noise`. Both pinned-digest DiT experts now
have confirmed REFIT-capable counterparts in the isolated `trtwan_engines_refit_test/` cache dir
(never touched the production pin). Next step: wire actual LoRA delta computation +
`refitter.set_named_weights()` + `refit_cuda_engine()` into `comfyui/nodes/dit_loader.py` (currently
a no-op for LoRA) and confirm a real LoRA visibly changes output in an actual ComfyUI workflow --
structural validation (names match) is necessary but not sufficient; the applied numerical effect
is still unverified.
