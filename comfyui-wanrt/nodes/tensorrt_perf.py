"""Self-contained ComfyUI loader nodes that combine model loading with optional SageAttention3
(Blackwell FP4) + MagCache optimization, applied to the returned `MODEL` object rather than the
checkpoint on disk.

Drag-and-drop: copy this one file into any `custom_nodes/*/` package's node list. No dependency
on anything else in this repo -- only `torch`, `numpy`, and ComfyUI's own `comfy`/`folder_paths`
modules (all already present in a ComfyUI install). `sageattn3` (SageAttention3) is a real
third-party dependency, but it is only ever imported lazily and, if missing, pip-installed
on-demand the first time `SageAttention` is actually enabled -- never at import time, never if the
user leaves it disabled.

## Why SageAttention3, not SpargeAttention/SageAttention2

This project's target GPU is Blackwell (RTX PRO 6000 Blackwell, sm_120). `thu-ml/SpargeAttn` has
no Blackwell kernel path at all -- verified directly against its `setup.py`
(`SUPPORTED_ARCHS = {8.0, 8.6, 8.7, 8.9, 9.0}`, Ampere through Hopper only) and confirmed by a real
failed build attempt on this GPU (2026-08-15). `thu-ml/SageAttention`'s `sageattention3_blackwell`
subproject, by contrast, is a Blackwell-*native* FP4 kernel with a dedicated `-gencode` branch for
`sm_100`/`sm_120`/`sm_121` -- also verified directly against its `setup.py`. It hard-errors on any
other compute capability, so this node only supports Blackwell; there is no fallback to
SpargeAttention or plain SageAttention2 for older GPUs. Note SageAttention3's own README says it
hasn't been validated as lossless on every model family (Wan isn't in its explicitly-tested list)
-- it's FP4-quantized, more aggressive than int8 sparse kernels, so watch output quality and prefer
the `MagCache`-only path if SageAttention3 visibly degrades a given generation.

## How SageAttention is patched

ComfyUI's attention dispatcher (`comfy.ldm.modules.attention.wrap_attn`) checks
`transformer_options["optimized_attention_override"]` on every attention call and, if present,
calls `override(original_backend_fn, q, k, v, heads, ...)` instead of the normal backend -- this
is the officially supported, model-local hook (added specifically so nodes don't have to globally
monkeypatch `comfy.ldm.wan.model.optimized_attention` the way some older custom nodes do). Setting
this key on a `model.clone()`'s `model_options["transformer_options"]` therefore patches attention
for *that* `MODEL` object only: `ModelPatcher.clone()` deep-copies `model_options`
(`comfy.utils.deepcopy_list_dict`), so two clones with different SageAttention/MagCache settings
never share the override, and nothing at the `torch`/module level is ever touched. Our override
falls back to the original backend function (passed in as `func`) whenever the FP4 kernel's real
constraints aren't met (explicit attention mask, head_dim not in {64, 128}, sequence < 128 tokens,
or any runtime error), so it never crashes generation -- same eager-fallback convention as
`vae_rt.py` in this package.

## How MagCache is patched

MagCache maintains an EMA-style running error estimate across diffusion steps and skips a block's
forward pass (reusing a cached residual) once the estimated accumulated error is still below a
threshold. The Wan2.2 forward pass and per-model magnitude-ratio calibration tables here are
ported from the official `Zehong-Ma/ComfyUI-MagCache` implementation (verified against its current
`nodes.py`), not reinvented. One deliberate change from upstream: upstream stores its mutable
skip/error/residual state as attributes on the *shared* `diffusion_model` nn.Module, which two
`MODEL` clones with different MagCache settings could in principle stomp on if run interleaved.
Here that same state instead lives inside `model_options["transformer_options"]["magcache_state"]`
-- a plain dict -- which `ModelPatcher.clone()` deep-copies just like everything else in
`model_options`, so it is genuinely independent per `MODEL` object without changing the caching
math itself. The state is additionally reset to its initial (no-skip) values whenever the sampler
wrapper detects `current_step_index == 0` (matched against `transformer_options["sample_sigmas"]`,
same detection upstream uses), so a new generation never reuses a previous generation's residual.

## LoRA / composability

Both patches are applied via `model_options`/`set_model_unet_function_wrapper`, never by mutating
weights, so `Load LoRA` after this loader still works normally (LoRA patches are applied by
ComfyUI's own weight-patch system, independent of `model_options`). SageAttention and MagCache can
also both be enabled together: MagCache's unet-function wrapper only swaps `forward_orig` for the
duration of one forward call (via `unittest.mock.patch.multiple`, restored immediately after), and
SageAttention's attention override only intercepts the innermost attention call -- neither depends
on or interferes with the other.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from typing import Any, Callable, Optional

import numpy as np
import torch

CATEGORY = "TensorRT-RT/Perf"
LOG_PREFIX = "[TensorRT-RT Perf]"


# --------------------------------------------------------------------------------------------
# Dependency management -- lazy install, sys.executable so it lands in ComfyUI's own env.
# --------------------------------------------------------------------------------------------

# thu-ml/SageAttention's Blackwell-native FP4 kernel lives in a subdirectory of that repo -- pip
# supports installing straight from a subdirectory via the `#subdirectory=` git URL fragment,
# verified against sageattention3_blackwell/README.md's own `cd sageattention3_blackwell &&
# python setup.py install` instructions (equivalent to installing that subdirectory as its own
# package). Import name is `sageattn3` (its setup.py's `PACKAGE_NAME`), not `sageattention3`.
_SAGE3_IMPORT_NAME = "sageattn3"
_SAGE3_PIP_SPEC = "git+https://github.com/thu-ml/SageAttention.git#subdirectory=sageattention3_blackwell"


def _pip_install(pip_spec: str, env: Optional[dict] = None) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pip", "install", pip_spec]
    print(f"{LOG_PREFIX} running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    full = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode == 0:
        return True, full[-2000:]
    # A failed CUDA extension build's pip/ninja output can run to tens of thousands of characters,
    # with the actual compiler error (a "fatal error:"/"error:" line) buried well before the final
    # few KB -- a plain tail has twice now been the difference between a real diagnosis and a
    # useless generic "Error compiling objects for extension" (root-caused both times only by
    # re-running the build directly over SSH instead of trusting this truncated message). Surface
    # every line mentioning "error" first, then fall back to the tail for full context.
    error_lines = "\n".join(line for line in full.splitlines() if "error" in line.lower())
    log = f"--- lines containing 'error' ---\n{error_lines[-4000:]}\n\n--- last 4000 chars of full output ---\n{full[-4000:]}"
    return False, log


def _cuda_extension_include_env() -> dict:
    """Env for building a CUDA extension against a CUDA toolkit that's missing dev headers pip-
    installed torch/nvidia-*-cu12 packages ship separately (confirmed real cause of a SageAttention3
    build failure on a RunPod 'slim' CUDA 12.8 image, 2026-08-15: `/usr/local/cuda/include` has no
    `cusparse.h` at all -- only `nvidia-cusparse-cu12`'s own pip package does, at
    `.../nvidia/cusparse/include/cusparse.h` -- but torch's own `ATen/cuda/CUDAContext.h` `#include
    <cusparse.h>`, so anything that transitively includes it fails to compile there). Adding every
    `site-packages/nvidia/*/include` dir to `CPATH` (respected by both gcc and nvcc's host-compiler
    pass) fixes this without touching the extension's own setup.py -- verified with a real build on
    that pod. No-op (falls back to the current environment) if the `nvidia` namespace package isn't
    importable, e.g. a non-pip-wheel torch install that doesn't need this at all.
    """
    env = dict(os.environ)
    try:
        import nvidia

        nvidia_dir = os.path.dirname(nvidia.__file__)
    except ImportError:
        return env

    include_dirs = sorted(glob.glob(os.path.join(nvidia_dir, "*", "include")))
    if not include_dirs:
        return env

    existing = env.get("CPATH", "")
    env["CPATH"] = os.pathsep.join(include_dirs + ([existing] if existing else []))
    return env


def ensure_sageattn3():
    """Import `sageattn3`, installing it into this Python env on first use.

    Raises RuntimeError with a clear, actionable message if install or import fails -- never
    silently falls back, per this node's explicit-enable contract.
    """
    import importlib

    try:
        return importlib.import_module(_SAGE3_IMPORT_NAME)
    except ImportError:
        pass

    print(f"{LOG_PREFIX} SageAttention3 (Blackwell FP4) not installed")
    print(f"{LOG_PREFIX} Installing SageAttention3 (compiles CUTLASS-based CUDA kernels; first run can take several minutes)...")

    ok, log = _pip_install("ninja")
    if not ok:
        raise RuntimeError(f"{LOG_PREFIX} ERROR: could not install the 'ninja' build dependency.\n\nInstallation error:\n{log}")

    ok, log = _pip_install(_SAGE3_PIP_SPEC, env=_cuda_extension_include_env())
    if not ok:
        raise RuntimeError(
            f"{LOG_PREFIX} ERROR: SageAttention3 could not be installed.\n\n{_gpu_summary()}\n\nInstallation error:\n{log}"
        )

    try:
        module = importlib.import_module(_SAGE3_IMPORT_NAME)
    except ImportError as exc:
        raise RuntimeError(f"{LOG_PREFIX} ERROR: SageAttention3 installed but failed to import: {exc}") from exc

    print(f"{LOG_PREFIX} SageAttention3 loaded successfully")
    return module


# --------------------------------------------------------------------------------------------
# GPU / CUDA compatibility
# --------------------------------------------------------------------------------------------

# thu-ml/SageAttention's sageattention3_blackwell/setup.py has a dedicated Blackwell FP4 CUTLASS
# kernel path, but only for the exact (major, minor) pairs it has an explicit `-gencode` branch
# for; any other compute capability hits that setup.py's own `else: raise RuntimeError("Unsupported
# GPU")`. Verified live against its current setup.py source, not assumed.
_SAGE3_SUPPORTED_CAPABILITIES = {(10, 0), (12, 0), (12, 1)}  # sm_100 (B200) / sm_120 (RTX PRO 6000 Blackwell, RTX 50-series) / sm_121


def _gpu_summary() -> str:
    if not torch.cuda.is_available():
        return "GPU: none detected (torch.cuda.is_available() is False)"
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return f"GPU: {name}\nCUDA: {torch.version.cuda}\nPyTorch: {torch.__version__}\nCompute capability: sm_{major}{minor}"


def check_sage3_gpu_compat() -> None:
    """Raise a clear RuntimeError if this GPU isn't one of the exact Blackwell compute
    capabilities `sageattention3_blackwell` builds for. There is no non-Blackwell fallback: this
    node only supports SageAttention3.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(f"{LOG_PREFIX} ERROR: SageAttention requires an NVIDIA GPU with CUDA; none detected.")

    capability = torch.cuda.get_device_capability(0)
    if capability not in _SAGE3_SUPPORTED_CAPABILITIES:
        major, minor = capability
        raise RuntimeError(
            f"{LOG_PREFIX} ERROR: SageAttention3 (Blackwell FP4) does not support this GPU (sm_{major}{minor}).\n\n"
            f"{_gpu_summary()}\n\n"
            f"sageattention3_blackwell only builds for {sorted(_SAGE3_SUPPORTED_CAPABILITIES)} (B200/RTX PRO 6000 "
            "Blackwell/RTX 50-series). This node has no fallback to SpargeAttention or SageAttention2 for other "
            "GPUs.\n\nWhat to do: disable SageAttention on this GPU."
        )


# --------------------------------------------------------------------------------------------
# SageAttention3 -- model-local attention override (comfy's `optimized_attention_override` hook)
# --------------------------------------------------------------------------------------------

SAGEATTN_MODES = ("Disabled", "Enabled")


class _AttentionSkip(Exception):
    """Internal control-flow signal: this call's shape/mask isn't supported by the FP4 kernel."""


def build_sageattn3_override() -> Callable:
    check_sage3_gpu_compat()
    sage3 = ensure_sageattn3()
    sageattn3_blackwell = sage3.sageattn3_blackwell
    warned = {"done": False}

    def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        try:
            if mask is not None:
                raise _AttentionSkip("explicit attention mask")
            if skip_reshape:
                b, _h, seq, dim_head = q.shape
                qh, kh, vh = q, k, v
            else:
                b, seq, dim_head = q.shape
                dim_head //= heads
                qh = q.view(b, seq, heads, dim_head).transpose(1, 2)
                kh = k.view(b, k.shape[1], heads, dim_head).transpose(1, 2)
                vh = v.view(b, v.shape[1], heads, dim_head).transpose(1, 2)
            if dim_head not in (64, 128):
                raise _AttentionSkip(f"unsupported head_dim={dim_head}")
            if seq < 128:
                raise _AttentionSkip(f"sequence too short ({seq} < 128)")

            out = sageattn3_blackwell(qh.contiguous(), kh.contiguous(), vh.contiguous(), is_causal=False)
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            return out
        except _AttentionSkip:
            pass
        except Exception as exc:  # never crash generation over an optimization -- fall back and log once.
            if not warned["done"]:
                warned["done"] = True
                print(f"{LOG_PREFIX} SageAttention3 call failed ({exc}); falling back to normal attention for the rest of this run.")

        return func(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

    return override


def apply_sageattn(model, mode: str):
    if mode == "Disabled":
        return model

    override = build_sageattn3_override()
    print(f"{LOG_PREFIX} SageAttention enabled -- backend: SageAttention3 (Blackwell FP4)")

    new_model = model.clone()
    to = new_model.model_options.setdefault("transformer_options", {})
    to["optimized_attention_override"] = override
    return new_model


# --------------------------------------------------------------------------------------------
# MagCache -- Wan2.2 magnitude-ratio calibration tables, ported verbatim from
# Zehong-Ma/ComfyUI-MagCache (nodes.py, SUPPORTED_MODELS_MAG_RATIOS), not reinvented.
# --------------------------------------------------------------------------------------------

WAN2_2_MAG_RATIOS: dict[str, np.ndarray] = {
    "wan2.2_t2v_14B": np.array([1.0] * 2 + [1.00124, 1.00155, 0.99822, 0.99851, 0.99696, 0.99687, 0.99703, 0.99732, 0.9966, 0.99679, 0.99602, 0.99658, 0.99578, 0.99664, 0.99484, 0.9949, 0.99633, 0.996, 0.99659, 0.99683, 0.99534, 0.99549, 0.99584, 0.99577, 0.99681, 0.99694, 0.99563, 0.99554, 0.9944, 0.99473, 0.99594, 0.9964, 0.99466, 0.99461, 0.99453, 0.99481, 0.99389, 0.99365, 0.99391, 0.99406, 0.99354, 0.99361, 0.99283, 0.99278, 0.99268, 0.99263, 0.99057, 0.99091, 0.99125, 0.99126, 0.65523, 0.65252, 0.98808, 0.98852, 0.98765, 0.98736, 0.9851, 0.98535, 0.98311, 0.98339, 0.9805, 0.9806, 0.97776, 0.97771, 0.97278, 0.97286, 0.96731, 0.96728, 0.95857, 0.95855, 0.94385, 0.94385, 0.92118, 0.921, 0.88108, 0.88076, 0.80263, 0.80181]),
    "wan2.2_ti2v_5B": np.array([1.0] * 2 + [0.99505, 0.99389, 0.99441, 0.9957, 0.99558, 0.99551, 0.99499, 0.9945, 0.99534, 0.99548, 0.99468, 0.9946, 0.99463, 0.99458, 0.9946, 0.99453, 0.99408, 0.99404, 0.9945, 0.99441, 0.99409, 0.99398, 0.99403, 0.99397, 0.99382, 0.99377, 0.99349, 0.99343, 0.99377, 0.99378, 0.9933, 0.99328, 0.99303, 0.99301, 0.99217, 0.99216, 0.992, 0.99201, 0.99201, 0.99202, 0.99133, 0.99132, 0.99112, 0.9911, 0.99155, 0.99155, 0.98958, 0.98957, 0.98959, 0.98958, 0.98838, 0.98835, 0.98826, 0.98825, 0.9883, 0.98828, 0.98711, 0.98709, 0.98562, 0.98561, 0.98511, 0.9851, 0.98414, 0.98412, 0.98284, 0.98282, 0.98104, 0.98101, 0.97981, 0.97979, 0.97849, 0.97849, 0.97557, 0.97554, 0.97398, 0.97395, 0.97171, 0.97166, 0.96917, 0.96913, 0.96511, 0.96507, 0.96263, 0.96257, 0.95839, 0.95835, 0.95483, 0.95475, 0.94942, 0.94936, 0.9468, 0.94678, 0.94583, 0.94594, 0.94843, 0.94872, 0.96949, 0.97015]),
    "wan2.2_i2v_14B": np.array([1.0] * 2 + [0.99191, 0.99144, 0.99356, 0.99337, 0.99326, 0.99285, 0.99251, 0.99264, 0.99393, 0.99366, 0.9943, 0.9943, 0.99276, 0.99288, 0.99389, 0.99393, 0.99274, 0.99289, 0.99316, 0.9931, 0.99379, 0.99377, 0.99268, 0.99271, 0.99222, 0.99227, 0.99175, 0.9916, 0.91076, 0.91046, 0.98931, 0.98933, 0.99087, 0.99088, 0.98852, 0.98855, 0.98895, 0.98896, 0.98806, 0.98808, 0.9871, 0.98711, 0.98613, 0.98618, 0.98434, 0.98435, 0.983, 0.98307, 0.98185, 0.98187, 0.98131, 0.98131, 0.9783, 0.97835, 0.97619, 0.9762, 0.97264, 0.9727, 0.97088, 0.97098, 0.96568, 0.9658, 0.96045, 0.96055, 0.95322, 0.95335, 0.94579, 0.94594, 0.93297, 0.93311, 0.91699, 0.9172, 0.89174, 0.89202, 0.8541, 0.85446, 0.79823, 0.79902]),
}

MAGCACHE_MODES = ("Disabled", "Fast", "Balanced", "Quality", "Custom")

# Below this many total sampler steps, a real generation (e.g. LightX2V-distilled 4-12 step Wan2.2)
# packs far more denoising work into each step than the mag_ratios calibration curve (captured on a
# ~38-76 step standard trajectory) assumes -- skip decisions become unreliable and can visibly wreck
# output. Confirmed via real user report: MagCache turning output brown starting at step 3 of a
# 4-12-step-per-phase LightX2V run, 2026-08-15. Not a hard block (some users may know what they're
# doing), just a loud one-time warning -- see the unet_wrapper_function in apply_magcache below.
_MAGCACHE_LOW_STEP_WARN_THRESHOLD = 20

_MAGCACHE_PRESETS: dict[str, dict[str, Any]] = {
    "Fast": dict(magcache_thresh=0.12, magcache_K=4, retention_ratio=0.15, start_step=0, end_step=-1),
    "Balanced": dict(magcache_thresh=0.06, magcache_K=2, retention_ratio=0.20, start_step=0, end_step=-1),
    "Quality": dict(magcache_thresh=0.03, magcache_K=1, retention_ratio=0.25, start_step=0, end_step=-1),
}

_WAN_VARIANT_HINTS = (
    ("ti2v", "wan2.2_ti2v_5B"),
    ("_5b", "wan2.2_ti2v_5B"),
    ("i2v", "wan2.2_i2v_14B"),
    ("t2v", "wan2.2_t2v_14B"),
)


def _magcache_params(mode: str, custom: Optional[dict[str, Any]]) -> dict[str, Any]:
    if mode == "Custom":
        return dict(custom or {})
    return dict(_MAGCACHE_PRESETS[mode])


def infer_wan_variant(filename: str) -> Optional[str]:
    name = filename.lower()
    for hint, variant in _WAN_VARIANT_HINTS:
        if hint in name:
            return variant
    return None


def is_wan_model(diffusion_model: Any) -> bool:
    return type(diffusion_model).__module__.startswith("comfy.ldm.wan")


def _fresh_magcache_state() -> dict[int, dict[str, Any]]:
    return {
        0: dict(skip_forward=False, accumulated_ratio=1.0, accumulated_err=0.0, accumulated_steps=0, residual_cache=None),
        1: dict(skip_forward=False, accumulated_ratio=1.0, accumulated_err=0.0, accumulated_steps=0, residual_cache=None),
    }


def _magcache_wanmodel_forward(self, x, t, context, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
    """Ported from Zehong-Ma/ComfyUI-MagCache's `magcache_wanmodel_forward`, with cache state read
    from `transformer_options["magcache_state"]` (isolated per MODEL clone) instead of an attribute
    on the shared diffusion_model nn.Module -- see module docstring."""
    import comfy.model_management as mm
    from comfy.ldm.wan.model import sinusoidal_embedding_1d

    magcache_thresh = transformer_options["magcache_thresh"]
    magcache_K = transformer_options["magcache_K"]
    cond_or_uncond = transformer_options["cond_or_uncond"]
    enable_magcache = transformer_options.get("enable_magcache", False)
    cur_step = transformer_options["current_step"]
    mag_ratios = transformer_options["mag_ratios"]
    magcache_state = transformer_options["magcache_state"]

    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})

    def update_cache_state(cache, step_index):
        if not enable_magcache:
            return
        cur_scale = mag_ratios[step_index]
        cache["accumulated_ratio"] *= cur_scale
        cache["accumulated_steps"] += 1
        cache["accumulated_err"] += float(np.abs(1 - cache["accumulated_ratio"]))
        if cache["accumulated_err"] <= magcache_thresh and cache["accumulated_steps"] <= magcache_K:
            cache["skip_forward"] = True
        else:
            cache["skip_forward"] = False
            cache["accumulated_ratio"] = 1.0
            cache["accumulated_steps"] = 0
            cache["accumulated_err"] = 0.0

    b = int(len(x) / len(cond_or_uncond))
    for i, k in enumerate(cond_or_uncond):
        update_cache_state(magcache_state[k], cur_step * 2 + i)

    skip_forward = False
    if enable_magcache:
        for k in cond_or_uncond:
            skip_forward = skip_forward or magcache_state[k]["skip_forward"]

    if skip_forward:
        for i, k in enumerate(cond_or_uncond):
            residual = magcache_state[k]["residual_cache"]
            if residual is not None:
                x[i * b:(i + 1) * b] += residual.to(x.device)
    else:
        ori_x = x.clone()
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)}

                out = blocks_replace[("double_block", i)](
                    {"img": x, "txt": context, "vec": e0, "pe": freqs},
                    {"original_block": block_wrap, "transformer_options": transformer_options},
                )
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
        for i, k in enumerate(cond_or_uncond):
            magcache_state[k]["residual_cache"] = (x - ori_x)[i * b:(i + 1) * b].to(mm.unet_offload_device())

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    return x


def apply_magcache(model, filename: str, mode: str, custom: Optional[dict[str, Any]] = None):
    if mode == "Disabled":
        return model

    diffusion_model = model.get_model_object("diffusion_model")
    if not is_wan_model(diffusion_model):
        raise RuntimeError(
            f"{LOG_PREFIX} ERROR: MagCache is enabled but this model "
            f"({type(diffusion_model).__module__}.{type(diffusion_model).__name__}) is not a supported Wan model."
        )

    variant = infer_wan_variant(filename)
    if variant is None or variant not in WAN2_2_MAG_RATIOS:
        raise RuntimeError(
            f"{LOG_PREFIX} ERROR: MagCache is enabled but couldn't determine the Wan2.2 variant "
            f"(t2v/i2v/ti2v) from filename '{filename}'. Supported: {sorted(WAN2_2_MAG_RATIOS)}."
        )

    params = _magcache_params(mode, custom)
    mag_ratios = torch.from_numpy(WAN2_2_MAG_RATIOS[variant]).float()

    new_model = model.clone()
    to = new_model.model_options.setdefault("transformer_options", {})
    to["magcache_thresh"] = params["magcache_thresh"]
    to["magcache_K"] = params["magcache_K"]
    to["retention_ratio"] = params["retention_ratio"]
    to["start_step"] = params.get("start_step", 0)
    to["end_step"] = params.get("end_step", -1)
    to["mag_ratios"] = mag_ratios
    to["magcache_state"] = _fresh_magcache_state()

    diffusion_model = new_model.get_model_object("diffusion_model")
    bound_forward = _magcache_wanmodel_forward.__get__(diffusion_model, diffusion_model.__class__)
    warned_low_steps = {"done": False}

    def unet_wrapper_function(model_function, kwargs):
        from unittest.mock import patch

        input_ = kwargs["input"]
        timestep = kwargs["timestep"]
        c = kwargs["c"]
        sigmas = c["transformer_options"]["sample_sigmas"]

        matched = (sigmas == timestep[0]).nonzero()
        if len(matched) > 0:
            current_step_index = matched.item()
        else:
            current_step_index = 0
            for i in range(len(sigmas) - 1):
                # referenced from https://github.com/kijai/ComfyUI-KJNodes model_optimization_nodes.py
                if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                    current_step_index = i
                    break

        magcache_state = c["transformer_options"]["magcache_state"]
        if current_step_index == 0:
            # new generation/sample sequence -- never reuse a previous generation's cached residual.
            for state in magcache_state.values():
                state.update(skip_forward=False, accumulated_ratio=1.0, accumulated_err=0.0, accumulated_steps=0, residual_cache=None)

        total_infer_steps = len(sigmas) - 1

        if current_step_index == 0 and total_infer_steps < _MAGCACHE_LOW_STEP_WARN_THRESHOLD and not warned_low_steps["done"]:
            warned_low_steps["done"] = True
            print(
                f"{LOG_PREFIX} WARNING: MagCache is calibrated against a ~38-76 step Wan2.2 trajectory, "
                f"but this run has only {total_infer_steps} steps -- looks like a distilled/LoRA-accelerated "
                "schedule (e.g. LightX2V). Each real step here does far more denoising work than the "
                "calibration assumes, so MagCache's skip decisions are likely wrong here and can visibly "
                "degrade output (washed-out/brown results). Recommend MagCache=Disabled for step counts "
                f"this low -- there's little to gain from it on an already-few-step schedule anyway."
            )
        start_step = c["transformer_options"]["start_step"]
        end_step = c["transformer_options"]["end_step"]
        if end_step < 0:
            end_step = total_infer_steps + end_step
        retention_ratio = c["transformer_options"]["retention_ratio"]

        if current_step_index >= int(total_infer_steps * retention_ratio) and start_step <= current_step_index <= end_step:
            c["transformer_options"]["enable_magcache"] = True
        else:
            c["transformer_options"]["enable_magcache"] = False

        calibration_len = len(c["transformer_options"]["mag_ratios"]) // 2
        if total_infer_steps == calibration_len:
            c["transformer_options"]["current_step"] = current_step_index
        else:
            # interpolate when the sampler's step count doesn't match the calibration length
            c["transformer_options"]["current_step"] = int(current_step_index * ((calibration_len - 1) / max(total_infer_steps - 1, 1)))

        with patch.multiple(diffusion_model, forward_orig=bound_forward):
            return model_function(input_, timestep, **c)

    new_model.set_model_unet_function_wrapper(unet_wrapper_function)
    print(f"{LOG_PREFIX} MagCache initialized ({variant}, mode={mode})")
    return new_model


# --------------------------------------------------------------------------------------------
# Shared node plumbing
# --------------------------------------------------------------------------------------------

# Ignored unless MagCache=="Custom" (see _magcache_params below) -- for Fast/Balanced/Quality these
# three values are irrelevant, and web/tensorrt_perf.js (if the whole comfyui-wanrt/ package is
# installed, not just this file) auto-fills them with the selected preset's real numbers so they
# never show a stale default that looks like it needs configuring.
_PERF_OPTIONAL_INPUTS = {
    "magcache_thresh": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.3, "step": 0.01, "tooltip": "MagCache Custom mode only -- ignored by Fast/Balanced/Quality. Max accumulated error before a skip streak is forced to recompute."}),
    "magcache_K": ("INT", {"default": 2, "min": 0, "max": 6, "step": 1, "tooltip": "MagCache Custom mode only -- ignored by Fast/Balanced/Quality. Max consecutive skipped steps."}),
    "magcache_retention_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.9, "step": 0.01, "tooltip": "MagCache Custom mode only -- ignored by Fast/Balanced/Quality. Fraction of early steps that always run dense."}),
}


def _apply_perf(model, model_filename: str, sageattn_mode: str, magcache_mode: str, magcache_thresh: float, magcache_K: int, magcache_retention_ratio: float):
    magcache_custom = {"magcache_thresh": magcache_thresh, "magcache_K": magcache_K, "retention_ratio": magcache_retention_ratio, "start_step": 0, "end_step": -1}

    model = apply_sageattn(model, sageattn_mode)
    model = apply_magcache(model, model_filename, magcache_mode, magcache_custom)

    print(
        f"{LOG_PREFIX}\n"
        f"Model: {model_filename}\n"
        f"SageAttention: {'ENABLED (SageAttention3, Blackwell FP4)' if sageattn_mode != 'Disabled' else 'disabled'}\n"
        f"MagCache: {'ENABLED (' + magcache_mode + ')' if magcache_mode != 'Disabled' else 'disabled'}\n"
        f"{_gpu_summary()}"
    )
    return model


class TensorRTDiffusionLoader:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "diffusion_model": (folder_paths.get_filename_list("diffusion_models"),),
                # Same three options/behavior as stock UNETLoader's weight_dtype -- verified
                # against the actual nodes.py this ComfyUI install runs, not assumed.
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"advanced": True}),
                "SageAttention": (list(SAGEATTN_MODES), {"default": "Disabled"}),
                "MagCache": (list(MAGCACHE_MODES), {"default": "Disabled"}),
            },
            "optional": dict(_PERF_OPTIONAL_INPUTS),
        }

    def load(self, diffusion_model, weight_dtype, SageAttention, MagCache, magcache_thresh=0.06, magcache_K=2, magcache_retention_ratio=0.2):
        import comfy.sd
        import folder_paths

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", diffusion_model)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        model = _apply_perf(model, diffusion_model, SageAttention, MagCache, magcache_thresh, magcache_K, magcache_retention_ratio)
        return (model,)


class TensorRTCheckpointLoader:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "SageAttention": (list(SAGEATTN_MODES), {"default": "Disabled"}),
                "MagCache": (list(MAGCACHE_MODES), {"default": "Disabled"}),
            },
            "optional": dict(_PERF_OPTIONAL_INPUTS),
        }

    def load(self, checkpoint, SageAttention, MagCache, magcache_thresh=0.06, magcache_K=2, magcache_retention_ratio=0.2):
        import comfy.sd
        import folder_paths

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint)
        model, clip, vae = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings")
        )[:3]
        model = _apply_perf(model, checkpoint, SageAttention, MagCache, magcache_thresh, magcache_K, magcache_retention_ratio)
        return (model, clip, vae)


NODE_CLASS_MAPPINGS = {
    "TensorRTDiffusionLoader": TensorRTDiffusionLoader,
    "TensorRTCheckpointLoader": TensorRTCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorRTDiffusionLoader": "TensorRT Diffusion Loader (SageAttention3 + MagCache)",
    "TensorRTCheckpointLoader": "TensorRT Checkpoint Loader (SageAttention3 + MagCache)",
}
