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
- [x] **Gray-fill-padding discrepancy fixed for the standalone API.** Fetched
      `WanFirstLastFrameToVideo.execute()` (`comfy_extras/nodes_wan.py`) directly from upstream
      ComfyUI source and ported its real algorithm exactly:
      `api/wan_engine.py`'s new `_build_image_to_video_conditioning` gray-fills (pixel value 0.5,
      `0.0` in this repo's `[-1,1]` convention) the entire target-length pixel video, overwrites
      real first/last frames, then VAE-encodes the whole padded video in one `encode_video` call
      (so padding latent frames reflect the VAE's real causal-conv response to nearby real frames
      instead of being exactly zero) and builds the mask at raw-frame granularity before
      reshaping into 4 channels — reproducing a real, confirmed **asymmetry**: a single first
      frame marks all 4 mask channels known at latent index 0 (aligns with a whole causal group),
      but a single last frame only marks 1-of-4 channels known at the last latent index. `dit_engine.py`
      gained a `ConditioningKind.IMAGE_VIDEO` fast path that concatenates this prebuilt pair
      directly, bypassing the old zero-pad placement logic. **Scope note:** the ComfyUI node-graph
      path (`comfyui/nodes/vae_encoder.py`'s `TensorRTVAEEncoder`, one node per frame with no
      visibility into target video length) still can't build this and keeps the old zero-pad
      approximation via `_concat_image_conditioning` — would need a joint node mirroring
      `WanFirstLastFrameToVideo`'s single-node signature to fix. CPU-only shape/value tests added
      (`tests/test_image_conditioning.py`, no GPU/model needed) verifying the asymmetric mask and
      both dit_engine branches.
- [x] **`vae_encoder` T>1 TensorRT build limitation — pivoted to a per-frame design instead of
      root-causing the build failure.** `vae_encoder`'s TensorRT build fails with a "Could not find
      any implementation" autotuner error on a `ForeignNode` (plausibly the VAE's bottleneck
      self-attention, whose sequence length scales with `H*W`, interacting with the multi-chunk
      cross-chunk-merge logic that only exists at `T>1`). Bisected across four (frames, resolution)
      points — 9@256x256 works (matches last night), 9@480x832 fails, 21@256x256 fails, 1@480x832
      works — a real complexity/size threshold scaling with *both* frame-count and resolution
      jointly, not either alone; the project's actual target config (81 frames @ 480x832) is well
      past it. Rather than root-cause the TensorRT failure itself (verbose/polygraphy per-node
      output, adjusting `_decompose_attention_for_export()` for the multi-chunk path, build-flag
      tuning — none attempted), `_build_image_to_video_conditioning` was rewritten to only ever
      need the one config already proven to build: `T=1`. It now calls `encode_image` once per
      *distinct* pixel content (gray, optionally first, optionally last — never once per output
      latent frame) and reuses each result across every latent position with that content,
      entirely avoiding the need for a large-`T` engine. **Correction to this bullet's earlier
      framing:** initially described this as losing "cross-chunk causal blending," implying a
      generation-quality cost — that conflated VAE-encode-time context with generation-time
      temporal consistency, which is entirely the DiT's own full self-attention across the whole
      sequence during denoising and is unaffected by how the input conditioning was encoded. The
      real difference is narrower: padding-position latent *values* no longer pick up a faint trace
      of a neighboring real frame the way the real algorithm's single whole-video encode would
      produce; the `mask` channel already tells the DiT which positions are authoritative regardless.
      Expected minor/second-order, consistent with the channel-order bug (the actual cause of every
      incoherent-output attempt) having dwarfed this from the start. Also drops the real algorithm's
      raw-frame-granularity mask asymmetry (a side effect specific to that joint chunked encode) —
      now simply all-4-channels-known at a real frame's latent index, all-4-channels-unknown
      elsewhere. Tests updated (`tests/test_image_conditioning.py`). See
      docs/wan2.2_i2v_14b_notes.md's corrected section for full detail.
- [x] **Real cache-key collision bug found and fixed while investigating the above.**
      `CacheKey.optimization_profile` was a profile *name* string, not the exporter's actual
      traced shape — two builds of the same component/profile-name but different exporter-kwargs
      (`frames=1` vs `frames=9`) silently overwrote each other's cache entry. Confirmed for real:
      a `frames=1` test build clobbered last night's working `frames=9` engine. Fixed via
      `ModelExporter.shape_digest()` (`export/base.py`) + `CacheKey.input_shape_digest`
      (`runtime/cache.py`), wired into both real `CacheKey` construction sites. Regression test:
      `tests/test_cache.py::test_different_input_shape_is_a_different_cache_entry`. **Any engine
      cached before this fix has ambiguous shape provenance and should be treated as untrustworthy
      until rebuilt.** Also surfaced, not fixed: `cli/commands/build.py`'s `run_engine` duplicates
      `export/pipeline.py`'s `run_export_pipeline` logic instead of calling it, despite that
      module's docstring claiming the CLI and the ComfyUI builder node share this path — they've
      diverged.
- [x] **first_frame=index-0/last_frame=index-(-1) temporal convention independently confirmed**
      against the same real source: `image[:start_image.shape[0]] = start_image` /
      `image[-end_image.shape[0]:] = end_image`. Matches what this repo already had.
- [x] Ran the full pipeline for real for the first time (2026-08-07): all four engines built,
      assembled a `model_dir`, ran `WanEngine.generate()` end to end (480x832, 81 frames, 30 steps).
      Found and fixed four more real, previously-undetected bugs along the way, all confirmed via
      an eager-vs-TensorRT byte-identical-input comparison (same methodology as the first session's
      decisive channel-order check):
      1. `comfy/ldm/flux/math.py`'s `rope()` mixes `float32`/`float64` in one `Einsum` — fine under
         eager's implicit promotion, fatal once `torch.export` freezes it. Fixed via a
         `_rope_fp32` monkeypatch on `comfy.ldm.flux.layers.rope` (not `wan.model.rope` —
         `EmbedND.forward()`, called unconditionally every DiT forward pass, resolves the name
         against its own module).
      2. `WanEngine._initial_latents()` defaulted to `float32` (`torch.randn` with no `dtype=`).
      3. `_build_image_to_video_conditioning`'s `mask` used `dtype=reference.dtype` (the caller's
         raw pixel dtype) instead of `dtype=image_latent.dtype` (the engine's actual output dtype).
      4. The text encoder's `text_embeds` *output* binding came out `float32` despite a `fp16`
         build — `_validate_precision` (`export/trt_build.py`) only ever checked network *inputs*,
         never outputs; a T5 implementation detail (plausibly an fp32-for-stability LayerNorm)
         slipped through undetected. Fixed both the specific case (`_TextEncoderWrapper` now casts
         its return value explicitly) and the general gap (`_validate_precision` now checks
         `network.get_output(i)` too).
      Bugs 2-4 share a pattern: none of them *error* — `torch.cat` silently promotes to the wider
      dtype rather than raising, so nothing surfaced until eager's strict dtype checking (or,
      for bug 4, this specific validation gap) exposed them.
      **After all four fixes: eager PyTorch produces completely clean output (zero NaN, sane
      stats) on the exact same inputs the built TensorRT DiT engine still returns 100% NaN for.**
      This conclusively isolates the remaining bug to the export/TensorRT-build pipeline itself —
      not conditioning construction, which is now fully validated correct. Not yet localized
      further (would need layer-by-layer activation comparison, not another single dtype-mismatch
      hunt) — stopped to report back given the time invested. See
      docs/wan2.2_i2v_14b_notes.md's latest session entries for full detail on each fix and the
      final decisive comparison.
- [x] Bisected the eager-vs-TensorRT divergence via activation-dump instrumentation (12 extra ONNX
      graph outputs, evenly spaced, auto-skipping fp32-by-design islands): first NaN appears at
      ~25% through the graph, on the main image-token stream, right around block 5's
      self-attention. Cross-referencing the raw ONNX ops found the likely cause: `load_dit()`
      never applied `_decompose_attention_for_export()` (unlike `load_text_encoder`/
      `_load_wan_vae`) — a *deliberate* decision from session 1 ("the DiT's own attention already
      finds a dedicated fused kernel with no such error"), but that observation was made at a tiny
      768-token smoke-test scale, not today's real ~32,760-token target — the same
      shape-dependent-TensorRT-correctness-gap pattern already proven for `vae_encoder` this
      session. Applied the fix to `load_dit()` too; rebuilding/retesting — not yet confirmed. Also
      ruled out: `comfy.ops.RMSNorm`'s dynamic-cast path (checked directly on the loaded model,
      inactive — plain native `torch.nn.RMSNorm` is what actually runs). Added
      `TRTWAN_BUILDER_OPT_LEVEL` (`export/trt_build.py`) along the way — cuts iteration rebuild
      time roughly in half for debugging, never for a real deployment build (defaults to max
      quality, `5`, explicitly). See docs/wan2.2_i2v_14b_notes.md's latest entries.
- [x] **Root cause found and fix confirmed this session: build the DiT in `bf16`, not `fp16`.**
      Full bisection chain: confirmed the attention-decomposition monkeypatch genuinely lands in
      the exported graph but TensorRT still returns 100% NaN regardless (it re-fuses the decomposed
      form back into essentially the same kernel — confirmed independently via an 88GiB OOM when
      debug-tapping those nodes forces defusion); `STRICT_NANS` made no difference; **eager PyTorch
      at the real ~32,760-token scale with native fused attention and the same trivial input came
      back completely clean**, proving the model math itself is correct and this is TensorRT-kernel
      -specific; bisected the NaN to inside block 0's self-attention specifically (not the
      modulation/gate path). Tested query-chunked attention (breaking TensorRT's fusion
      pattern-match) combined with `bf16` — clean. Isolated the two: **`fp16` + chunking alone is
      still 100% NaN; `bf16` alone (no chunking) is also completely clean** (`nan_frac=0.0`,
      realistic finite output matching the eager reference) — so `bf16`'s wider dynamic range is
      the actual fix, not the chunking/fusion theory. Landed two real code changes:
      `ModelExporter.dtype` (`export/base.py`) now follows `TRTWAN_LOADER_DTYPE` instead of being
      hardcoded fp16 (needed for `bf16` builds to even be internally consistent). Query-chunking
      was tried, confirmed unnecessary (isolated: `fp16`+chunked still NaN'd, `bf16` alone without
      chunking didn't), and stripped back out per user decision. Hardened against silent
      regression: `load_dit()` and `DiTExporter.dtype` now hardcode `bf16` unconditionally and
      ignore `TRTWAN_LOADER_DTYPE` (warn if set to anything else) — unlike every other loader,
      which still follows it normally. **Promoted to the real (non-debug) `trtwan_engines/` cache
      and confirmed clean** at full `builder_optimization_level=5`. Also found, not yet fixed:
      `CacheKey`'s `input_shape_digest` can't detect a traced-graph change that doesn't alter
      declared input shapes — two different graphs silently shared one cache slot during this
      session's isolation testing. See wan2.2_i2v_14b_notes.md's "Isolation: bf16 alone fixes it"
      and "Fix promoted to production" entries for full detail. Still open, unrelated: the VAE
      5D-unification assumption in `export/exporters/vae.py` remains unverified against real
      ComfyUI source.
- [ ] **New, separate bug found running the first-ever full `generate()`:** with the DiT fix
      promoted and text_encoder/vae_encoder/vae_decoder built fresh, ran a real I2V generation
      (81 frames @ 832x480, the real default target, `close_green_chair_start/end.png`,
      `scripts/run_i2v_generate.py`) — no crash, no NaN (confirms the DiT fix holds up under real
      end-to-end use, not just isolated trivial-input checks), but the decoded output video is
      **pure noise, no coherent structure, no resemblance to the input images.** Not yet diagnosed:
      skimmed `FlowMatchEulerScheduler` (`scheduler/flow_match.py`) and `DiTEngine.denoise_step`'s
      CFG formula (`engine/dit_engine.py`) and both look structurally correct at a glance (standard
      shifted-sigma Euler integration, standard `uncond + scale*(cond-uncond)` CFG) — needs a real
      bisection, not a read-through fix.

      **Update, same session: implemented real Wan 2.2 MoE two-pass expert switching** (`WanEngine`
      now takes `dit_high_noise`/`dit_low_noise`, switches at 50% of `num_inference_steps` — switch
      rule confirmed against ComfyUI's own official blueprints, not assumed; see
      wan2.2_i2v_14b_notes.md) and **built the `low_noise` expert engine** — mechanically confirmed
      working (log shows the switch firing at the right step, output distribution genuinely
      changes) but **did not fix the noise problem**. Also **directly tested the VAE round-trip in
      isolation** (encode a real image, immediately decode it back, no DiT/scheduler involved) —
      **fully coherent, clearly recognizable** — this rules the VAE out entirely.

      **Both DiT and VAE are now individually cleared; the bug is elsewhere.** Real remaining gap:
      every DiT correctness check so far (eager-vs-TensorRT, the bf16 isolation, the MoE
      confirmation) used **`timestep=0`** — never a representative sample, just a convenient
      input-independent NaN probe. Real generation sweeps `timestep≈1000` down to `0`; nobody has
      checked eager-vs-TensorRT agreement at a realistic high-noise timestep with real random
      latents. Also still unconfirmed: text embedding sanity (tokenizer mismatch — `google/umt5-xxl`
      vs ComfyUI's own SentencePiece tokenizer, flagged in `runpod_setup.md`, never independently
      checked). **Next session should start with an eager-vs-TensorRT DiT comparison at a realistic
      high-`timestep` sample**, the largest remaining untested gap. See wan2.2_i2v_14b_notes.md's
      "MoE switch works mechanically but doesn't fix coherence" entry
- [ ] Build engines for the default resolution profiles and confirm cache hit/miss behavior —
      also revisit `_build_optimization_profile`: found while doing this that
      `ResolutionProfile.height`/`.width` are never actually read there, so multiple resolution
      profiles currently just build identical duplicate optimization profiles. **Real target,
      clarified 2026-08-07: not a fixed list of named profiles — arbitrary width/height divisible
      by 16 (Wan's hard requirement), e.g. 32x32, 480x832, 720x1088, 960x1248.** Current shape
      comes entirely from `DiTExporter`'s hardcoded `latent_frames`/`latent_height`/`latent_width`
      kwargs (default 21/60/104, i.e. 81 frames @ 480x832) via a single static-or-narrow-dynamic
      optimization profile — nothing today lets a caller pick a resolution at generation time
      within one built engine, let alone arbitrary /16 values. Real fix needs the DiT's dynamic
      axes (`dynamic_axes()`, `export/exporters/dit.py`) actually driven by desired min/max
      width/height (not just the current opt-only fixed shape), `_build_optimization_profile`
      wired to read `ResolutionProfile.height`/`.width` instead of ignoring them, and
      `vae_encoder`/`vae_decoder`'s already-known joint frame/resolution TensorRT build ceiling
      (see the `vae_encoder` `frames=1` note above) re-examined against whatever range gets chosen
      — not started
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
      negligible quality loss) — `runtime/precision.py`'s `select_precision` docstring promises
      this ("FP8 where a per-op quality check clears it") but the implementation has no such gate:
      `auto` unconditionally maps Blackwell -> `"fp8"` ceiling. Combined with the loader always
      casting to fp16 regardless of requested precision, and `_validate_precision`'s real gap
      below, this silently produced a genuinely-fp16 engine mislabeled "fp8" in the cache — hit
      for real on 2026-08-06 building `text_encoder` with `--precision auto` on this Blackwell
      box; see docs/wan2.2_i2v_14b_notes.md's 2026-08-06 session section. **Until this gate is
      real, always pass `--precision fp16` explicitly — never rely on `auto` on Blackwell.**
- [ ] `_validate_precision`'s real gap, found via the above: it only checks the ONNX graph's
      *network input* tensors, not internal weights — so a component whose inputs happen to be
      all-non-float (text_encoder's `input_ids`/`attention_mask` are both `INT64`) has zero float
      tensors to check and the validator silently no-ops, regardless of requested precision. Only
      components with float inputs (dit's `x`/`context`, vae's `pixels`/`latent`) actually get
      checked, and correctly *reject* an fp8 build attempt there today (confirmed: `STRONGLY_TYPED`
      networks have no `BuilderFlag.FP16`/`BF16`/`FP8` at all — precision comes entirely from the
      ONNX graph's own tensor dtypes, and nothing in the export pipeline casts to fp8 or inserts
      calibrated Q/DQ nodes). Needs a real PTQ/calibration pass (e.g. TensorRT Model Optimizer)
      inserted before ONNX export before "fp8" as selected by `runtime/precision.py`
      can actually work — see wan2.2_i2v_14b_notes.md
- [ ] `examples/loaders/wan_comfyui_loader.py`'s `load_dit()` force-casts to `TRTWAN_LOADER_DTYPE`
      (default fp16) — fine for this fp16 checkpoint, but would silently clobber an
      already-quantized (fp8/int4/AWQ) checkpoint's precision. Should detect and preserve the
      checkpoint's native dtype instead of always casting
- [ ] `EngineCache`'s content-addressed `{digest}.engine`/`.json` filenames (`runtime/cache.py`)
      make on-disk debugging harder than it needs to be — a rebuild of the same component just
      lands under a new opaque hash, old ones pile up needing manual cleanup, and there's no way to
      tell what's what from `ls` alone. User feedback (2026-08-07): prefer human-readable filenames
      (e.g. `{component}.engine`, overwritten in place on rebuild) over content-addressed ones —
      easier to debug, and a rebuild just overwrites rather than accumulating stale entries. Not
      done yet, deliberately deferred (mid-debugging-session, didn't want to spend the time then).
      Trade-off to think through before implementing: content-addressing is what makes
      `EngineCache.get()`'s cache-hit/miss check possible without re-running the build — a
      human-readable scheme needs a different mechanism for that (e.g. a separate lookup index
      mapping cache-key -> filename) if the cache-hit behavior should be kept.

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
