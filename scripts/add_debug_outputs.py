"""Add extra graph outputs (debug taps) to an ONNX model, evenly spaced across node indices.

CPU-only, `load_external_data=False` -- doesn't need the (tens of GB) external weights file, only
touches graph structure/metadata. Saves the new .onnx next to the original external-data file so
the external-data reference still resolves (same relative filename, same directory).

Skips any candidate whose inferred dtype isn't genuinely fp16 (e.g. `patch_embedding`'s own conv
output, RMSNorm-internal fp32 stability computations) by walking forward to the next fp16 tensor --
TensorRT's STRONGLY_TYPED `_validate_precision` check (`export/trt_build.py`) rejects fp32 graph
outputs outright, so a naive tap would just fail the build instead of producing useful data.

Usage: python3 add_debug_outputs.py <onnx_in> <onnx_out> <names_out.txt> [n_taps] [range_start] [range_end]
"""
import sys

import onnx
from onnx import TensorProto, helper, shape_inference

onnx_in, onnx_out, names_out = sys.argv[1], sys.argv[2], sys.argv[3]
n_taps = int(sys.argv[4]) if len(sys.argv) > 4 else 12
range_start = int(sys.argv[5]) if len(sys.argv) > 5 else None
range_end = int(sys.argv[6]) if len(sys.argv) > 6 else None

model = onnx.load(onnx_in, load_external_data=False)
inferred = shape_inference.infer_shapes(model, strict_mode=False)

dtype_by_name = {vi.name: vi.type.tensor_type.elem_type for vi in inferred.graph.value_info}
for vi in list(inferred.graph.output) + list(inferred.graph.input):
    dtype_by_name[vi.name] = vi.type.tensor_type.elem_type

nodes = model.graph.node
n = len(nodes)
lo = 0 if range_start is None else range_start
hi = n if range_end is None else min(range_end, n)
step = max((hi - lo) // n_taps, 1)
picked = []
for k in range(n_taps):
    idx = min(lo + k * step, hi - 1)
    while idx < n:
        fp16_outputs = [o for o in nodes[idx].output if dtype_by_name.get(o) == TensorProto.FLOAT16]
        if fp16_outputs:
            picked.append((idx, fp16_outputs[0]))
            break
        idx += 1

existing_outputs = {o.name for o in model.graph.output}
for idx, name in picked:
    if name in existing_outputs:
        continue
    model.graph.output.append(helper.make_tensor_value_info(name, dtype_by_name[name], None))
    existing_outputs.add(name)

onnx.save(model, onnx_out)

with open(names_out, "w") as f:
    for idx, name in picked:
        f.write(f"{idx}\t{name}\n")

print(f"graph has {n} nodes; added {len(picked)} debug outputs (requested {n_taps})")
for idx, name in picked:
    print(f"  idx={idx} name={name}")
