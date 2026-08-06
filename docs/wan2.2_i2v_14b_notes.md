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
