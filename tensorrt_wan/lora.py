"""Shared LoRA/DiT naming knowledge: which submodules LoRA touches, and how to recover their
ONNX/TensorRT weight-initializer names from an exported DiT graph.

Used at build time (`export.trt_build`, to mark weights individually refittable) and at inference
time (`engine.lora_refit`, to know which TensorRT weight name a LoRA checkpoint key maps to) --
lives here, not in either of those, so neither has to import the other just for this.
"""

from __future__ import annotations

import json
from pathlib import Path

LORA_SUBMODULES = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)


def onnx_weight_name_map(onnx_path: str | Path) -> dict[tuple[int, str], str]:
    """Walk `onnx_path`'s graph to find the ONNX initializer name backing each LoRA-relevant
    weight matrix (q/k/v/o projections + FFN up/down, per block), keyed by `(block_idx, submodule)`.

    torch.export's decomposition transposes every `nn.Linear.weight` before it feeds `MatMul`,
    which strips the original `blocks.{i}.{submodule}.weight` parameter path and replaces it with
    a synthetic name (e.g. `val_570`) tied to graph position, not source name. Biases keep their
    real names, though, and each bias's sole consumer is an `Add` whose other input is the `MatMul`
    output that used the weight -- so walking bias name -> Add -> sibling MatMul -> weight
    initializer recovers the mapping. Verified by hand against blocks 0, 5, and 39, then validated
    end-to-end against `trt.Refitter.get_all_weights()` on real REFIT-flagged builds of both DiT
    experts (400/400 exact match both times) -- see docs/wan2.2_i2v_14b_notes.md's 2026-08-08/09
    Refit-API entries.

    Graph-only (`load_external_data=False`): doesn't touch the multi-GB of actual weight values.
    Stops at the first block index with zero matches, so it self-limits to however many blocks the
    model actually has rather than assuming 40.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph
    init_names = {init.name for init in graph.initializer}
    producer = {output: node for node in graph.node for output in node.output}

    mapping: dict[tuple[int, str], str] = {}
    block_idx = 0
    while True:
        found_any = False
        for submodule in LORA_SUBMODULES:
            bias_name = f"blocks.{block_idx}.{submodule}.bias"
            consumers = [n for n in graph.node if bias_name in n.input]
            if len(consumers) != 1 or consumers[0].op_type != "Add":
                continue
            add_node = consumers[0]
            other = next(x for x in add_node.input if x != bias_name)
            matmul_node = producer.get(other)
            if matmul_node is None or matmul_node.op_type != "MatMul":
                continue
            weight_candidates = [x for x in matmul_node.input if x in init_names]
            if len(weight_candidates) != 1:
                continue
            mapping[(block_idx, submodule)] = weight_candidates[0]
            found_any = True
        if not found_any:
            break
        block_idx += 1
    return mapping


def onnx_weight_names(onnx_path: str | Path) -> list[str]:
    """Flat list of `onnx_weight_name_map`'s values, in block/submodule order."""
    mapping = onnx_weight_name_map(onnx_path)
    return [mapping[(block_idx, submodule)] for block_idx, submodule in sorted(mapping)]


def weight_map_path_for_engine(engine_path: str | Path) -> Path:
    """Sidecar JSON path convention for an engine's weight-name map -- next to the `.engine` file,
    same stem, `.lora_map.json` suffix. A fixed, derivable-from-just-the-engine-path convention so
    callers (the ComfyUI LoRA node) never need a separate `onnx_path` input to find it.
    """
    return Path(engine_path).with_suffix(".lora_map.json")


def save_weight_name_map(mapping: dict[tuple[int, str], str], path: str | Path) -> None:
    """Persist `onnx_weight_name_map`'s result as JSON, surviving the onnx file's deletion (it's
    routinely deleted after a build to save disk -- see docs/wan2.2_i2v_14b_notes.md). Keys are
    joined `"{block_idx}.{submodule}"` since JSON has no tuple keys.
    """
    Path(path).write_text(json.dumps({f"{block_idx}.{submodule}": name for (block_idx, submodule), name in mapping.items()}, indent=2))


def load_weight_name_map(path: str | Path) -> dict[tuple[int, str], str]:
    """Inverse of `save_weight_name_map`."""
    raw = json.loads(Path(path).read_text())
    mapping: dict[tuple[int, str], str] = {}
    for key, name in raw.items():
        block_str, submodule = key.split(".", 1)
        mapping[(int(block_str), submodule)] = name
    return mapping
