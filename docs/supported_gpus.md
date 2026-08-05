# Supported GPUs

Architecture detection is in [`runtime/gpu.py`](../tensorrt_wan/runtime/gpu.py); precision
defaults per architecture are in [`runtime/precision.py`](../tensorrt_wan/runtime/precision.py)
(see [optimization_strategy.md](optimization_strategy.md)).

| Architecture | Compute capability | Default precision | Status |
|---|---|---|---|
| Blackwell | sm_100, sm_120 | FP8 (fallback FP16) | Primary validation target (RTX PRO 6000 Blackwell) |
| Hopper | sm_90 | FP16 | Supported |
| Ada Lovelace | sm_89 | FP16 | Supported |
| Ampere | sm_80, sm_86 | FP16 | Supported |
| Turing | sm_75 | FP16 | Supported, untested |
| Volta | sm_70 | FP16 | Supported, untested |
| Pascal | sm_60, sm_61 | FP32 | Supported, untested; no FP16 Tensor Cores |

An architecture not in this table (or a driver too old to enumerate compute capability) maps to
`GPUArchitecture.UNKNOWN` and defaults to FP16 rather than failing — see `_classify()` in
`runtime/gpu.py`.

## Checking what's detected

```bash
trtwan gpu-report
```

Reports every visible device's name, architecture, compute capability, memory, and the precision
that would be selected. Works without TensorRT installed (reports TensorRT as unavailable) and
without any GPU present (reports zero devices) — see
[`runtime/manager.py`](../tensorrt_wan/runtime/manager.py)'s `RuntimeManager.diagnostics()`.

## Validation status

Everything above is `nvidia-smi`/`torch.cuda`-derived metadata logic, not yet exercised against
real hardware in this repository (see PLAN.md's development rule). The RTX PRO 6000 Blackwell
RunPod instances are the primary validation target — see [roadmap.md](roadmap.md).
