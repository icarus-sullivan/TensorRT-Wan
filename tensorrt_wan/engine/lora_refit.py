"""Apply a LoRA's weight delta directly to a REFIT-capable TensorRT DiT engine via TensorRT's
Refit API, since ComfyUI's stock LoRA nodes can't work on a TensorRT-backed model (see
comfyui/nodes/dit_loader.py's `TensorRTDiTModule` -- it has no real per-block weights for a normal
`ModelPatcher` patch to land on; a LoRA loaded through the stock node silently lands on a dummy
Conv3d and never touches computation, confirmed empirically before this module existed).

Two real Wan LoRA key conventions exist in the wild (see docs/wan2.2_i2v_14b_notes.md's
2026-08-08 Refit-API entry for the full survey):
  - `diffusion_model.blocks.{i}.{submodule}.lora_down.weight` / `.lora_up.weight` (rank R)
  - `diffusion_model.blocks.{i}.{submodule}.lora_A.weight` / `.lora_B.weight` (down=A, up=B, same
    `delta = scale*(up@down)` math)
Both are handled identically here. `.diff_b` (bias delta), `.diff` (norm delta), `.diff_m`
(modulation delta) keys are logged as present-but-unsupported rather than silently dropped --
only the 400 q/k/v/o/ffn *weight* matrices are marked refittable on our engines (see
`export.trt_build`), biases/norms/modulation are baked in at build time and can't be refit yet.

Not yet supported: composing multiple LoRAs in one engine. Each `apply_lora()` call recomputes
from the *original* checkpoint's base weight + this LoRA's delta and refits fresh -- it does not
read back and add to whatever's currently refit into the engine (TensorRT's `Refitter` can't
reliably read back a never-yet-refit weight, see docs/wan2.2_i2v_14b_notes.md), so calling this
twice in a row applies the second LoRA only, it does not stack the two.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from tensorrt_wan.engine.base import TensorRTEngineWrapper
from tensorrt_wan.lora import LORA_SUBMODULES, load_weight_name_map
from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)

_KEY_RE = re.compile(
    r"^diffusion_model\.blocks\.(?P<block>\d+)\.(?P<submodule>.+)\.(?P<kind>lora_down|lora_up|lora_A|lora_B)\.weight$"
)
_UNSUPPORTED_SUFFIXES = (".diff_b", ".diff_m", ".diff")


def compute_lora_deltas(lora_sd: dict[str, torch.Tensor], strength: float) -> dict[tuple[int, str], torch.Tensor]:
    """Group a LoRA state dict's `lora_down`/`lora_up` (or `lora_A`/`lora_B`) pairs by
    `(block_idx, submodule)` and compute each one's `strength * (up @ down)` delta, in the
    checkpoint's native `[out, in]` layout (not yet transposed to TensorRT's `[in, out]`).
    """
    down_up: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    unsupported_seen: set[str] = set()

    for key, tensor in lora_sd.items():
        if any(key.endswith(suffix) for suffix in _UNSUPPORTED_SUFFIXES):
            unsupported_seen.add(key)
            continue
        m = _KEY_RE.match(key)
        if not m:
            continue
        block_idx = int(m.group("block"))
        submodule = m.group("submodule")
        slot = "down" if m.group("kind") in ("lora_down", "lora_A") else "up"
        down_up.setdefault((block_idx, submodule), {})[slot] = tensor

    if unsupported_seen:
        logger.warning(
            "LoRA has %d bias/norm/modulation delta key(s) this engine can't refit yet (only "
            "q/k/v/o/ffn weight matrices are marked refittable) -- applying weight deltas only, "
            "these keys will have no effect: %s",
            len(unsupported_seen),
            sorted(unsupported_seen)[:10],
        )

    deltas: dict[tuple[int, str], torch.Tensor] = {}
    for (block_idx, submodule), pair in down_up.items():
        if "down" not in pair or "up" not in pair:
            raise ValueError(
                f"LoRA key for block {block_idx} {submodule!r} has only one of down/up present "
                f"({list(pair)}) -- malformed LoRA file."
            )
        down, up = pair["down"].float(), pair["up"].float()
        deltas[(block_idx, submodule)] = strength * (up @ down)
    return deltas


def _read_safetensors_header(path: str | Path) -> dict:
    """Parse just a safetensors file's header (8-byte little-endian length + that many bytes of
    JSON) -- gives every tensor's `data_offsets` without touching any actual tensor data. Cheap
    even against this project's network-mounted `/workspace` (a few KB, not 28GB).
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))


def _load_all_base_weights(checkpoint_path: str | Path) -> dict[tuple[int, str], torch.Tensor]:
    """Load every LoRA-relevant base weight from `checkpoint_path`, reading tensors in ascending
    file-offset order via `safe_open`'s lazy/mmap'd `get_tensor()` -- not `load_file()`.

    Confirmed necessary on real hardware, the hard way: the original per-key approach (400
    `get_tensor()` calls in `(block_idx, submodule)` order, which doesn't match on-disk order) was
    slow over this project's network-mounted `/workspace` (MooseFS). The fix that replaced it
    (`load_file()`, reading the *entire* ~28GB checkpoint into real anonymous RAM) was faster but
    wrong in a worse way: this pod's container has a 175GB cgroup memory ceiling (not the host's
    full 1.5TB, confirmed via `/sys/fs/cgroup/memory.max` after a real crash), and loading two
    experts' full 28GB checkpoints alongside ComfyUI/VAE/CLIP/text-encoder pushed past it --
    `memory.events` showed `oom_kill: 1`, a silent SIGKILL with no Python traceback. Reading in
    file-offset order via `safe_open` gets the same "make it sequential, not scattered" benefit
    without ever materializing the whole file: mmap'd pages are file-backed (reclaimable by the OS
    under memory pressure), not anonymous memory that counts fully against the cgroup limit.

    Cached in bf16 (not fp32) on `TensorRTEngineWrapper._lora_base_weight_cache` -- halves the
    cache's steady-state memory (~28GB/expert instead of ~56GB), still comfortably survives the
    upcast-to-fp32-then-back-down round trip `apply_lora()` does for the actual delta math.
    """
    header = _read_safetensors_header(checkpoint_path)

    wanted: list[tuple[int, tuple[int, str], str]] = []  # (file_offset, cache_key, safetensors_name)
    block_idx = 0
    while True:
        found_any = False
        for submodule in LORA_SUBMODULES:
            name = f"blocks.{block_idx}.{submodule}.weight"
            if name in header:
                wanted.append((header[name]["data_offsets"][0], (block_idx, submodule), name))
                found_any = True
        if not found_any:
            break
        block_idx += 1
    wanted.sort(key=lambda entry: entry[0])

    weights: dict[tuple[int, str], torch.Tensor] = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        for _offset, cache_key, name in wanted:
            weights[cache_key] = f.get_tensor(name).to(torch.bfloat16)
    return weights


def apply_lora(
    wrapper: TensorRTEngineWrapper,
    checkpoint_path: str | Path,
    weight_map_path: str | Path,
    lora_path: str | Path,
    strength: float,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Compute `lora_path`'s delta against `checkpoint_path`'s base weights and refit `wrapper`'s
    engine with the result. `weight_map_path` is the `(block_idx, submodule) -> TensorRT weight
    name` sidecar JSON written at build time (see `tensorrt_wan.lora.save_weight_name_map`,
    `cli.commands.build`) -- deliberately not the onnx file itself, since that's routinely deleted
    after a build to save disk (see docs/wan2.2_i2v_14b_notes.md) and this needs to keep working
    without it.

    Base weights are read from `checkpoint_path` only on `wrapper`'s *first* LoRA application and
    cached on `wrapper` afterward (`TensorRTEngineWrapper._lora_base_weight_cache`) -- TensorRT's
    Refitter can't read a never-yet-refit weight back off the compiled engine itself (confirmed via
    NVIDIA's docs: `get_named_weights()` returns null + an error until something has been
    explicitly set), so the first read is unavoidable, but every subsequent LoRA/strength change on
    the same loaded model reuses the cache instead of touching the checkpoint again.

    `dtype` must match the engine's build precision (bf16 for every DiT engine this project has
    built so far) -- `TensorRTEngineWrapper.refit_weights()` validates this and fails loudly if
    wrong rather than silently casting, so a mismatch here surfaces immediately as an error, not
    as a silent quality regression.
    """
    name_map = load_weight_name_map(weight_map_path)
    lora_sd = load_file(str(lora_path))
    deltas = compute_lora_deltas(lora_sd, strength)

    missing_in_map = [key for key in deltas if key not in name_map]
    if missing_in_map:
        raise ValueError(
            f"LoRA references {len(missing_in_map)} (block, submodule) pair(s) not found in this "
            f"engine's weight map: {missing_in_map[:10]}... -- LoRA/engine architecture mismatch?"
        )

    if wrapper._lora_base_weight_cache is None:
        logger.info("Loading base weights from %s for LoRA refit (first use -- cached after this)", checkpoint_path)
        wrapper._lora_base_weight_cache = _load_all_base_weights(checkpoint_path)
    base_weights = wrapper._lora_base_weight_cache

    final_weights: dict[str, torch.Tensor] = {}
    for (block_idx, submodule), delta in deltas.items():
        if (block_idx, submodule) not in base_weights:
            raise ValueError(
                f"No base weight for block {block_idx} {submodule!r} in {checkpoint_path} -- "
                "checkpoint/LoRA architecture mismatch?"
            )
        base = base_weights[(block_idx, submodule)]  # cached bf16 -- upcast for the add, see below
        if base.shape != delta.shape:
            raise ValueError(
                f"Base weight for block {block_idx} {submodule!r} shape {tuple(base.shape)} != "
                f"LoRA delta shape {tuple(delta.shape)}."
            )
        # Checkpoint/LoRA are both PyTorch's [out, in] nn.Linear convention; TensorRT's weight
        # is the transposed [in, out] (torch.export decomposes Linear into a pre-transposed
        # MatMul input -- see tensorrt_wan.lora's docstring) -- transpose once, after summing.
        # base.float(): the cache stores bf16 (memory footprint, see _load_all_base_weights), but
        # the add itself happens in fp32 alongside delta (already fp32 from compute_lora_deltas)
        # for accuracy, then cast down to the engine's real dtype at the very end.
        new_weight = (base.float() + delta).to(dtype).T.contiguous()
        final_weights[name_map[(block_idx, submodule)]] = new_weight

    logger.info(
        "Applying LoRA %s (strength=%.3g) -- refitting %d weights", lora_path, strength, len(final_weights)
    )
    wrapper.refit_weights(final_weights)
