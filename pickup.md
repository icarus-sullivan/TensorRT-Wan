# Pickup notes — TensorRT-Wan

Last session ended: 2026-08-09, ~9:30am (pod shut off by user, mid-session). Local repo has
**uncommitted changes, not yet committed** (`git log -1` still shows `9aedb1b`):
```
 M comfyui/nodes/__init__.py
 M comfyui/nodes/dit_loader.py
 M docs/wan2.2_i2v_14b_notes.md
 M scripts/build_engines.sh
 M tensorrt_wan/cli/commands/build.py
 M tensorrt_wan/engine/base.py
 M tensorrt_wan/export/trt_build.py
 M tensorrt_wan/runtime/cache.py
?? comfyui/nodes/lora_loader.py
?? tensorrt_wan/engine/lora_refit.py
?? tensorrt_wan/lora.py
```
All present and safe locally regardless of pod state — nothing here depends on the pod surviving.

## What happened this session (long one)

1. **Cold-started a net-new pod** (old one from the prior session was dead), rebuilt the pinned
   known-working DiT engines from scratch. Along the way, fixed a real pre-existing bug in
   `scripts/build_engines.sh`: `RES="${WIDTH}x${HEIGHT}"` should have been
   `"${HEIGHT}x${WIDTH}"` (schema's `ResolutionProfile` names are `HxW`, not `WxH`) — this had
   silently broken every cold-start `build_engines.sh dit` run since it was written.
2. **Resolved the Refit-API LoRA "still open" question from the prior session**: torch.export
   strips weight-matrix parameter names during decomposition, but the recovery is a clean
   deterministic graph walk (bias name → `Add` → sibling `MatMul` → weight initializer), not a
   heuristic. Validated end-to-end on real REFIT-flagged builds of both DiT experts:
   `Refitter.get_all_weights()` returned exactly the 400 predicted names, zero discrepancies,
   twice (high and low noise). Full writeup in `docs/wan2.2_i2v_14b_notes.md`'s 2026-08-08/09
   Refit-API entries — read that before re-deriving any of this.
3. **Built a real LoRA application pipeline**: `tensorrt_wan/lora.py` (shared ONNX weight-name
   recovery + JSON sidecar save/load), `tensorrt_wan/engine/lora_refit.py` (LoRA delta computation
   + apply orchestration), `TensorRTEngineWrapper.refit_weights()` (engine/base.py), and a new
   ComfyUI node `TensorRTDiTLoraLoader` (comfyui/nodes/lora_loader.py) — chainable like a real LoRA
   loader (`model -> TensorRTDiTLoraLoader -> model`), auto-derives the checkpoint path and
   weight-name-map sidecar from the model, only asks for `lora_name`/`strength`.
4. **Registered a real ComfyUI dropdown** for engine selection: `ComfyUI/models/tensorrt_engines/`
   (symlink-aliased to wherever engines actually live, nothing copied) shows up as `engine_name` in
   `TensorRTDiTLoader` instead of a free-text path.
5. **Found and partially fixed a build-speed dead end**: initially misdiagnosed a ~22min gap as a
   `bytes(serialized)` copy cost; direct measurement showed it was actually only ~27s (real,
   applied fix, kept) — the 22min was almost certainly GPU/CPU contention from a concurrent ComfyUI
   test, not a code issue. Also found the timing cache does NOT meaningfully speed up an identical
   rebuild (~10s difference, not explained) — flagged as open, not resolved.
6. **First real end-to-end LoRA test crashed** — root cause chased through three iterations:
   - First it just hung. Diagnosed (via `/proc/<pid>/io`, thread states, since `py-spy` is blocked
     by this container's ptrace restrictions) as 400 individual `safe_open().get_tensor()` calls
     against the 28GB checkpoint over `/workspace`'s flaky MooseFS mount.
   - Fixed via `safetensors.torch.load_file()` (bulk sequential read) + caching the result on
     `TensorRTEngineWrapper._lora_base_weight_cache` so it only happens once per loaded model.
   - That "fix" **crashed the container** — confirmed via `/sys/fs/cgroup/memory.events` showing
     `oom_kill: 1`. This container has a **175GB memory limit** (not the host's 1.5TB) —
     `load_file()` eagerly materializes the *entire* 28GB checkpoint as real anonymous RAM, and
     doing that for both experts alongside ComfyUI/VAE/CLIP/text-encoder blew past it. Silent
     SIGKILL, no Python traceback — that's why it looked like a mysterious hang at first.
   - **Real fix (written, syntax-checked locally, NOT yet synced to the pod)**: back to
     `safe_open`'s lazy/mmap'd reads (OS-reclaimable pages, don't count fully against the cgroup
     limit), but now reading the ~400 needed tensors in **ascending file-offset order** (parsed
     directly from the safetensors header) instead of block/submodule order — that's what actually
     makes it sequential on-disk, independent of bulk-vs-per-key API choice. Also cache in bf16
     instead of fp32 (halves steady-state cache memory, ~28GB/expert instead of ~56GB).

## Next step when resuming — THE critical one

**The OOM fix in `tensorrt_wan/engine/lora_refit.py` has never been synced to the pod or tested.**
The pod currently (if it survived) still has the OLD buggy version that will OOM-crash again if you
retest the LoRA workflow without syncing first. Do this before anything else LoRA-related:

1. Confirm pod alive + `/workspace` state (see "Resume" sections below).
2. Sync `tensorrt_wan/lora.py`, `tensorrt_wan/engine/lora_refit.py`, `tensorrt_wan/engine/base.py`,
   `tensorrt_wan/cli/commands/build.py`, `tensorrt_wan/export/trt_build.py`,
   `tensorrt_wan/runtime/cache.py`, `comfyui/nodes/__init__.py`, `comfyui/nodes/dit_loader.py`,
   `comfyui/nodes/lora_loader.py` — both into the repo checkout AND the live
   `ComfyUI/custom_nodes/tensorrt_wan_comfyui/` mirror (rsync, per `deploy_comfyui_integration.sh`'s
   pattern — or just re-run that script for the `comfyui/` half).
3. Verify import cleanly in ComfyUI's own venv before trusting it:
   ```bash
   cd /workspace/runpod-slim/ComfyUI && .venv-cu128/bin/python -c "
   import sys; sys.path.insert(0, 'custom_nodes')
   import tensorrt_wan_comfyui
   print(list(tensorrt_wan_comfyui.NODE_CLASS_MAPPINGS.keys()))
   import folder_paths
   print(folder_paths.get_filename_list('tensorrt_engines'))
   "
   ```
4. **Restart ComfyUI yourself** (never do this for the user — hard rule, see memory) and retest:
   `TensorRTDiTLoader` (pick `dit_high_noise_refit.engine` or `dit_low_noise_refit.engine` from the
   dropdown) → `TensorRTDiTLoraLoader` (pick a lora, set strength) → sampler. Watch
   `/workspace/runpod-slim/ComfyUI/user/comfyui_8188.log` for
   `tensorrt_wan.engine.lora_refit: Loading base weights...` / `Applying LoRA...` lines to confirm
   it actually completes this time, and check `/sys/fs/cgroup/memory.events`'s `oom_kill` count
   hasn't incremented.
5. **The actual LoRA numerical/visual effect has never been confirmed** — structural validation
   (names match) passed twice, but no generation has ever completed with a LoRA applied. That's the
   real open question: does the output actually look different with the LoRA on vs off.
6. Also still unverified this session: the plain (non-LoRA) reference workflow on the *production*
   pinned engines (`mean=92.15`, green chair) — never explicitly re-confirmed after all of tonight's
   changes, though the REFIT test engines alone (no LoRA) reportedly generated 4 movies fine.

## Resume: same pod (state likely intact)

Old connection (may be dead — pod was shut off, not just idle):
```bash
ssh -p 11789 -i ~/.ssh/id_ed25519 root@205.196.144.18 "echo alive"
```
If that fails, get fresh connection details from the RunPod dashboard and go to "Cold start" below.

If it connects:
```bash
ssh -p 11789 -i ~/.ssh/id_ed25519 root@205.196.144.18 "
ls -la /workspace/runpod-slim/trtwan_known_working_engines/          # pinned non-refit engines
ls -la /workspace/runpod-slim/trtwan_engines_refit_test/*.engine     # REFIT test engines
ls -la /workspace/runpod-slim/ComfyUI/models/tensorrt_engines/       # dropdown symlinks
"
```
If all present, skip straight to syncing the OOM fix (above) — no rebuild needed, both REFIT
engines are already validated and just need the corrected `lora_refit.py`.

## Cold start: net-new pod, nothing carried over

Follow `docs/runpod_setup.md` for the general playbook. Specific to where this session left off:
- The two pinned non-refit engines and the two REFIT test engines (+ `.lora_map.json` sidecars) all
  live under `/workspace/runpod-slim/` — if the persistent volume survived, none of tonight's
  ~4 hours of engine builds need repeating.
- If the persistent volume did NOT survive: rebuild the pinned engines per the previous session's
  cold-start steps (now fixed — `scripts/build_engines.sh` no longer has the WxH/HxW bug), then
  rebuild the two REFIT engines with `TRTWAN_ENABLE_REFIT=1 TRTWAN_BUILDER_OPT_LEVEL=2` in an
  isolated cache dir (never the same dir as the pinned engines — `CacheKey.digest()` doesn't
  include the REFIT flag, so building into the same dir silently overwrites the pin under an
  identical filename). See `docs/wan2.2_i2v_14b_notes.md`'s Refit-API entries for the exact
  commands and why the isolation matters.
- Either way, sync the local repo (uncommitted changes, see top of this file) before doing anything
  LoRA-related — the pod won't have tonight's fixes otherwise.
