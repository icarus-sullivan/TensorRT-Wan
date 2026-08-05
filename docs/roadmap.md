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
- [ ] What's still not done: real I2V conditioning content (the 36-channel `x` was still zeros,
      not real noise+image+mask), the reshape-constant-baking correctness gap noted above, and
      ONNX export + engine build for text_encoder/vae_encoder/vae_decoder (only DiT verified)
- [ ] Build engines for the default resolution profiles and confirm cache hit/miss behavior —
      also revisit `_build_optimization_profile`: found while doing this that
      `ResolutionProfile.height`/`.width` are never actually read there, so multiple resolution
      profiles currently just build identical duplicate optimization profiles
- [ ] Run `WanEngine.generate()` end to end for T2V; compare output against the FP16 PyTorch
      reference
- [ ] Same for I2V
- [ ] Wire a real FlashAttention-2/3 or SageAttention backend into `CustomAttentionPlugin`
      (currently unimplemented, see `custom_attention/kernel_dispatch.cpp`)
- [ ] `RotaryEmbedding` plugin (`plugins/csrc/rotary_embedding/kernel.cu`) is confirmed wrong —
      implements rotate-half, but Wan actually uses interleaved-pair rotation (confirmed against
      ComfyUI's real `comfy/ldm/flux/math.py` source). Rewrite against
      `examples/loaders/wan_comfyui_loader.py`'s cloned `_apply_rope1` reference; see
      wan2.2_i2v_14b_notes.md and plugins.md's validation status section
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
