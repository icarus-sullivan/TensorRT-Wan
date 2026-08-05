# Optimization Strategy

## Precision selection

Implemented in [`runtime/precision.py`](../tensorrt_wan/runtime/precision.py). Rule: pick the
highest-performance precision that doesn't measurably degrade output quality; never drop
precision purely to save memory.

| Architecture | Default (`mode="auto"`) |
|---|---|
| Blackwell | FP8 (if `allow_fp8`), else FP16 |
| Hopper / Ada / Ampere / Turing / Volta | FP16 |
| Pascal | FP32 |

Every decision records a `reason` string (`PrecisionDecision.reason`) surfaced in
`trtwan gpu-report`/`optimization-report` and in logs — a silently-wrong precision choice is
exactly the kind of bug that needs an audit trail, not just a config value.

FP8 is offered where the architecture supports it, never forced: `allow_fp8=False` in
[`PrecisionConfig`](../tensorrt_wan/config/schema.py) always drops to FP16 regardless of
architecture. Per-op FP8 quality gating (only quantizing where a calibration pass shows no
measurable quality loss) is validation-phase work — see [roadmap.md](roadmap.md).

## Optimization profiles / resolution coverage

One TensorRT engine per component covers every configured `ResolutionProfile` via multiple
`IOptimizationProfile`s (see [`export/trt_build.py`](../tensorrt_wan/export/trt_build.py)) —
static engines for the common resolutions listed in PLAN.md, plus user-defined profiles via
config, without needing one engine file per resolution.

## Engine caching

[`EngineCache`](../tensorrt_wan/runtime/cache.py) keys on model hash + TensorRT version + CUDA
version + GPU architecture + optimization profile + precision. Any of those changing is a cache
miss — see [engine_generation.md](engine_generation.md#caching).

## GPU-resident scheduling

The sampling loop (`scheduler/`, `engine/dit_engine.py`'s `generate()`) keeps all per-step state
on-device and never syncs to host mid-loop — see [architecture.md](architecture.md#why-the-scheduler-is-gpu-resident).

## Automatic fallback

[`runtime/fallback.py`](../tensorrt_wan/runtime/fallback.py)'s `run_with_fallback` wraps every
engine call ([`engine/base.py`](../tensorrt_wan/engine/base.py)): a TensorRT failure (unsupported
op, shape, or precision) degrades to the equivalent PyTorch module, logs a warning, and never
crashes the run — the project's explicit fallback rule.

## What's not implemented yet

CUDA Graphs capture, kernel auto-selection tuning, and cross-stream overlap are PLAN.md
requirements not yet reflected in code — they depend on having real built engines to profile
against, which happens in the RunPod validation phase (see [roadmap.md](roadmap.md)).
