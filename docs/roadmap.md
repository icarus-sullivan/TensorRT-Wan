# Roadmap

## Phase 1 — Structure (this repository, current state)

Everything PLAN.md's development rule permits without a GPU: module structure, interfaces,
exporters, plugin source, CLI, ComfyUI nodes, config schema, docs, tests (unexecuted). No model
has been exported, no engine built, no inference run, nothing profiled or benchmarked, no
generated engine validated.

Done:

- [x] Core module structure (`runtime`, `conditioning`, `scheduler`, `engine`, `export`,
      `plugins`, `config`, `api`, `cli`)
- [x] `WanEngine` standalone Python API
- [x] `trtwan` CLI (gpu-report, cache, export, build, inspect, list, optimization-report)
- [x] ComfyUI custom node package (13 nodes)
- [x] TensorRT plugin source for 8 ops, with shared boilerplate
- [x] Unexecuted test suite
- [x] Documentation set

## Phase 2 — RunPod GPU validation (next)

On RTX PRO 6000 Blackwell instances:

- [ ] Run the unexecuted test suite; fix whatever doesn't hold once `torch`/`tensorrt` are
      actually present
- [ ] Build `libtensorrt_wan_plugins.so` (`scripts/build_plugins.sh`) and unit-test each plugin
      against its PyTorch reference op (see [plugins.md](plugins.md)'s validation status section)
- [x] Wire a real Wan checkpoint loader (`--loader` function) reusing ComfyUI's own model class —
      `examples/loaders/wan_comfyui_loader.py`, verified on RunPod against both Wan 2.2 14B
      experts; see [wan2.2_i2v_14b_notes.md](wan2.2_i2v_14b_notes.md)
- [x] `DiTExporter`/`DiTEngine` input naming (`x`/`timestep`/`context`, `in_channels` vs. the
      VAE's unrelated `latent_channels`) fixed to match the real `WanModel.forward()` signature,
      confirmed via a successful `torch.export.export()` against the real 14.29B-param checkpoint
- [ ] Still open: `example_inputs()`'s `x` traces against zeros, not a real channel-concatenated
      (noise + image-latent + mask) tensor — fine for T2V, not sufficient for a numerically
      correct I2V engine. Need to locate whatever ComfyUI node builds that 36-channel tensor
      normally (likely a stock `WanImageToVideo`-equivalent) and replicate its channel order/mask
      construction — see wan2.2_i2v_14b_notes.md's conditioning-mismatch section
- [x] `torch.onnx.export` proven against the real 14.29B-param DiT (needed `opset_version=23` for
      RMSNorm + bypassing `comfy_kitchen.apply_rope1`, see wan2.2_i2v_14b_notes.md)
- [x] Ran through the actual `trtwan export onnx` CLI / `DiTExporter` class (not just the
      standalone script) with real dynamic shapes covering frame-count/height/width. Found and
      fixed two more real bugs along the way: (1) all four `ModelExporter.example_inputs()`
      built tensors with no `device=`, defaulting to CPU against a GPU-resident model — added
      `ModelExporter.device`; (2) `torch.export.Dim(min=,max=)` is an assertion Wan's real
      patch-alignment arithmetic can't satisfy across a full range — switched to `Dim.AUTO`. Both
      fixes are in `export/base.py`/`export/torch_export.py`, not script-local hacks. Batch stays
      fixed at 1 even under `Dim.AUTO` (reasonable — one video per request); frame-count/height/
      width stayed properly dynamic. See wan2.2_i2v_14b_notes.md
- [x] TensorRT engine build proven against the real 14.29B-param DiT — 26.63 GiB engine, 118.4s
      build, on real Blackwell hardware (RTX PRO 6000). Found and fixed along the way: `pip
      install tensorrt` resolves the wrong CUDA runtime (needs `tensorrt-cu12` explicitly, not
      plain `tensorrt`), and `STRONGLY_TYPED` networks have no `BuilderFlag.FP16`/`BF16`/`FP8` at
      all in this TensorRT version — `export/trt_build.py`'s `_apply_precision_flags` (which
      would have crashed identically) is now `_validate_precision`. See wan2.2_i2v_14b_notes.md.
      **Full three-stage pipeline (`torch.export` → ONNX → TensorRT) now proven end to end.**
- [x] Re-ran the full pipeline through the *actual* `trtwan export onnx` / `trtwan build engine`
      CLI (not the standalone script) with real dynamic shapes (frame-count/height/width) via
      `DiTExporter`. Found and fixed three more real framework bugs in the process: all four
      `ModelExporter.example_inputs()` missing `device=`/`dtype=` (defaulting to CPU/fp32 against
      a GPU/fp16 model — silently "succeeded" through `torch.export`/ONNX, only caught by
      TensorRT's stricter parser), declaring batch as a dynamic profile axis when the model
      actually specializes it to a fixed value, and the engine cache defaulting to non-persistent
      `/root/.cache` with no CLI override (added a global `--cache-dir` flag). See
      wan2.2_i2v_14b_notes.md for all of these in detail — this is now the most-verified path in
      the whole repo.
- [x] `DiTEngine._build_inputs` (`engine/dit_engine.py`) now channel-concatenates
      `first_frame`/`last_frame` conditioning (+ a mask) onto `x` in the confirmed
      noise(16)++image_latent(16)++mask(4) order, instead of raising `NotImplementedError` for
      any non-text conditioning kind. Source-only fix (`_concat_image_conditioning`), no GPU here
      to run it. Zero-pads the reference frame's latent to `x`'s full temporal length and uses a
      binary (1.0 at the conditioned frame / 0.0 elsewhere) mask broadcast across all 4 mask
      channels — both are documented best-effort defaults, **not** confirmed to match ComfyUI's
      real `WanImageToVideo` node (which gray-pads pixel-space and VAE-encodes the whole padded
      video, likely producing non-zero latents for the padding frames). Needs a RunPod numeric
      comparison before trusting I2V output quality.
- [x] `VAEEncoderExporter` (`export/exporters/vae.py`) unified around a 5D `(B, 3, T, H, W)`
      `pixels` input with a dynamic frame axis (T=1 opt case), instead of a fixed rank-4
      `(B, 3, H, W)` input — fixes a real internal inconsistency: `VAEEncoderEngine.encode_video`
      (`engine/vae_engine.py`) was already calling the same built engine with a rank-5 tensor,
      which a rank-4-exported engine cannot accept. `encode_image` now unsqueezes to rank-5
      (T=1) before inference to match. Source-only fix, no GPU here to run it — rests on an
      unconfirmed assumption that Wan's real VAE module is video-native (causal 3D conv, image =
      T=1) rather than genuinely having two different forward paths; needs checking against
      ComfyUI's actual VAE source (not available in this environment) before trusting it.
- [x] RoPE fix re-verified end to end: full DiT `torch.export`→ONNX→TensorRT pipeline re-run on
      real Blackwell hardware with the fixed `RotaryEmbedding` kernel, still succeeds (26.6GiB
      engine). Found and fixed two more environment/version-skew bugs along the way (newer torch
      needs `dynamic_shapes` dict entries for every arg, not just dynamic ones; newer
      `torch.onnx` needs `onnxscript` installed separately) — see wan2.2_i2v_14b_notes.md's
      2026-08-06 session section.
- [x] `load_text_encoder`/`load_vae_encoder`/`load_vae_decoder` written (`wan_comfyui_loader.py`)
      — didn't exist before, only `load_dit` did.
- [x] **All four component engines now built** (DiT, text_encoder, vae_encoder, vae_decoder), all
      in `/workspace/runpod-slim/trtwan_engines/`. Text encoder and VAE both needed the same real
      fix: TensorRT 11.2's native-ONNX-`Attention`-op import path can't find a fused kernel for
      either (masked T5 self-attention *or* the VAE's unmasked bottleneck self-attention — not
      mask-specific), and `IAttention.decomposable` (the fix the error message itself suggests)
      isn't reachable from Python in this TensorRT version at all (confirmed: no downcast from
      the generic `ILayer`, no constructor). Real fix: monkeypatch
      `scaled_dot_product_attention` to a decomposed matmul+softmax+matmul form before export so
      the native op is never emitted — worked for both. VAE additionally needed a
      `cudnn_convolution`-has-no-FakeTensor-kernel fix (monkeypatch
      `comfy.ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND = False` for the export trace only, safe since
      no real cuDNN kernel runs during FakeTensor tracing) and a checkpoint correction —
      `wan2.2_vae.safetensors` is the *wrong* file for these 14B checkpoints (z_dim=48, for Wan
      2.2's separate 5B TI2V model); `wan_2.1_vae.safetensors` is correct (z_dim=16, matches the
      DiT). Also found and fixed a real `EngineCache` bug along the way: `CacheKey` had no
      `component` field, so `vae_encoder`/`vae_decoder` (same checkpoint, same profile/precision)
      collided on the same cache digest — a decoder build attempt was silently served the
      encoder's engine. See wan2.2_i2v_14b_notes.md's 2026-08-06 session section for full detail
      on all of the above.
- [x] First real end-to-end run, all four engines together, real prompt + two real reference
      images. Found and fixed two serious infra bugs in `engine/base.py`'s
      `TensorRTEngineWrapper._infer_trt`, both silent-corruption classes rather than one-off:
      (1) `context.set_input_shape()`'s bool return value was never checked — now raises loudly;
      (2) `set_tensor_address()` was handed raw pointers with no dtype conversion, so a float32
      `timestep` (scheduler's default) got byte-reinterpreted as float16 by the engine (built with
      a float16 `timestep` input), producing NaN on the very first denoising step — now every
      input is cast to the engine's own declared dtype before use. Also fixed a real
      `DiTEngine._build_inputs` bug caught before it ever ran on GPU: image conditioning was
      concatenated once per kind, so `first_frame`+`last_frame` together would have produced 56
      channels against the engine's fixed 36 — now built as one combined 16ch+4ch pair. VAE
      encode→decode round-trip independently verified correct (real recognizable chair image).
- [x] **Root cause of the content-quality bug narrowed conclusively.** Swept `shift` in
      `[1,2,3,5,8]` — ruled out, all values converge to nearly-identical (wrong) output. Then ran
      the decisive check: the built DiT TensorRT engine vs. the real eager 14.29B-param checkpoint
      on byte-identical inputs (`x`/`timestep`/`context`) — **cosine_similarity=0.999995,
      max_abs_diff=0.0137**, essentially a perfect match within fp16 noise. **This clears the
      entire export/build pipeline** (RoPE fix, decomposed attention, everything) — the engine
      faithfully reproduces the real model. Since eager and TensorRT agree almost exactly and
      still produce bad output together, the bug isn't engine conversion — it's what gets fed to
      the model. Every other candidate (image-conditioning magnitude, CFG scale, scheduler shift)
      is now ruled out too, leaving one clear remaining suspect: `_concat_image_conditioning`'s
      unconfirmed zero-padding/binary-mask policy vs. what ComfyUI's real `WanImageToVideo` node
      actually builds. See wan2.2_i2v_14b_notes.md's "Shift sweep and the decisive
      eager-vs-TensorRT comparison" section.
- [x] **Found and fixed the real bug — first genuinely coherent I2V output.** Read
      `comfy_extras/nodes_wan.py`'s real `WanImageToVideo` node and `WAN21.concat_cond`
      (`comfy/model_base.py`, the code that actually assembles the DiT's `x`): real channel order
      is `noise(16) ++ mask(4) ++ image_latent(16)` — mask *before* image latent.
      `_concat_image_conditioning` had them reversed since it was first written. Fixed
      (`engine/dit_engine.py`). Re-ran the real prompt/images end to end: `final_latents` went
      from mean=1.76/std=4.79 (runaway drift) to mean=0.05/std=1.10 (stable, well-behaved) —
      decoded frames now show real spatial structure (chair-shaped dark band against a wall with
      matching window/pipe detail), consistent across all 9 frames. Still low quality (20 steps,
      256×256 test res) and mask polarity/gray-fill-padding details remain only partially
      confirmed (see below), but structurally working for the first time. See
      wan2.2_i2v_14b_notes.md's "Found and fixed: real channel-order bug" section.
- [ ] Still open: the gray-fill-padding discrepancy (`WanImageToVideo` gray-fills pixel-space
      before VAE-encoding the whole padded video in one call — this repo zero-pads directly in
      latent space instead, likely smaller-magnitude than the channel-order bug was but
      unmeasured), independent confirmation of the first_frame=index-0/last_frame=index-(-1)
      temporal convention, a higher-step/full-resolution quality run, and the VAE 5D-unification
      assumption in `export/exporters/vae.py` — all still unverified against real ComfyUI source
- [ ] Build engines for the default resolution profiles and confirm cache hit/miss behavior —
      also revisit `_build_optimization_profile`: found while doing this that
      `ResolutionProfile.height`/`.width` are never actually read there, so multiple resolution
      profiles currently just build identical duplicate optimization profiles
- [ ] Run `WanEngine.generate()` end to end for T2V; compare output against the FP16 PyTorch
      reference
- [ ] Same for I2V
- [ ] Wire a real FlashAttention-2/3 or SageAttention backend into `CustomAttentionPlugin`
      (currently unimplemented, see `custom_attention/kernel_dispatch.cpp`)
- [x] `RotaryEmbedding` plugin (`plugins/csrc/rotary_embedding/kernel.cu`) rewritten from
      rotate-half to Wan's actual interleaved-pair rotation, matching
      `examples/loaders/wan_comfyui_loader.py`'s cloned `_apply_rope1` reference (adjacent pairs
      x[2i]/x[2i+1] rotated by a shared angle; cos/sin tables now expected repeat-interleaved,
      not concat-duplicated). Source-only fix, no GPU here to build/run it — still needs the
      isolated numeric comparison against the PyTorch reference on RunPod before it's trusted in
      a built engine; see plugins.md's validation status section
- [ ] Per-op FP8 quality gating on Blackwell (PLAN.md: never reduce precision without confirming
      negligible quality loss)
- [ ] Real FP8 quantization is not implemented. `export/trt_build.py`'s `_validate_precision`
      (confirmed against a real TensorRT 11.2 build: `STRONGLY_TYPED` networks have no
      `BuilderFlag.FP16`/`BF16`/`FP8` at all — precision comes entirely from the ONNX graph's own
      tensor dtypes) will correctly *reject* an fp8 build attempt today, since nothing in the
      export pipeline casts to fp8 or inserts calibrated Q/DQ nodes — it just doesn't silently
      build the wrong thing anymore. Needs a real PTQ/calibration pass (e.g. TensorRT Model
      Optimizer) inserted before ONNX export before "fp8" as selected by `runtime/precision.py`
      can actually work — see wan2.2_i2v_14b_notes.md
- [ ] `examples/loaders/wan_comfyui_loader.py`'s `load_dit()` force-casts to `TRTWAN_LOADER_DTYPE`
      (default fp16) — fine for this fp16 checkpoint, but would silently clobber an
      already-quantized (fp8/int4/AWQ) checkpoint's precision. Should detect and preserve the
      checkpoint's native dtype instead of always casting

## Phase 3 — Feature completeness

- [ ] ControlNet / IP-Adapter / LoRA conditioning sources exercised end to end (interfaces exist
      in `conditioning/sources/`, untested against real adapters)
- [ ] No "TensorRT LoRA Loader" ComfyUI node exists at all (`comfyui/nodes/` has the 13 nodes from
      PLAN.md's suggested list, none of which is a LoRA node — `TensorRTConditioningManager`'s
      `lora` socket has nothing to feed it). Also not just a missing node: semantics genuinely
      differ from a normal ComfyUI `LoraLoader → sampler` flow, since a TensorRT engine's weights
      are baked in at build time — there's no live weight-patching at inference the way eager
      PyTorch allows. A real LoRA workflow needs LoRA selection *before* `trtwan build engine`,
      not as a graph node in the generation workflow. `conditioning/sources/lora.py`/
      `engine/dit_engine.py` already assume this; nothing surfaces it as a usable ComfyUI flow yet
- [ ] `examples/comfyui_workflow_i2v.json`'s `EmptyLatentImage` placeholder has no frame-count
      control at all (it's a 4D image-latent node, not video — no length/frames widget exists on
      it). The Note node in that workflow already flags this; needs an actual 5D
      empty-video-latent node (with a real frame-count widget) built or wired in before the
      example workflow is anything more than a wiring proof-of-concept
- [ ] Dynamic height/width for the DiT may not be safely correct even where TensorRT accepts the
      build: a reshape downstream of `patch_embedding` was observed using a token-count constant
      baked from the `opt` example shape rather than one derived from the actual runtime input
      size (`Profile kMIN/kMAX values are not self-consistent` warnings, volumes off by exactly
      the opt/actual frame-count ratio). Needs a dedicated investigation — likely resolution:
      switch to `Dim.STATIC` per resolution profile (separate static engines, already a
      first-class supported strategy per PLAN.md) rather than chasing full dynamic H/W/T support
      for this architecture. See wan2.2_i2v_14b_notes.md
- [ ] Video-to-video and editing workflows
- [ ] CUDA Graphs capture for the sampling loop
- [ ] Multi-GPU / tensor-parallel inference (PLAN.md's future-expansion list)

## Phase 4 — Ecosystem

- [ ] REST/gRPC API
- [ ] Web UI
- [ ] Streaming/real-time generation
- [ ] Audio generation + audio/video sync, if/when Wan supports it

Dates are intentionally not attached to these phases — they're gated on GPU access and upstream
Wan/TensorRT releases, not a calendar.
