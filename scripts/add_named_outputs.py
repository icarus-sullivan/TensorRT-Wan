"""Add specific, named tensors as extra ONNX graph outputs (debug taps) -- unlike
add_debug_outputs.py's evenly-spaced auto-picker, this is for "I already know exactly which
tensors I want to inspect." Errors instead of skipping if a requested tensor isn't fp16 (the
caller asked for it by name deliberately, so silently substituting a neighbor would be wrong here).

Usage: python3 add_named_outputs.py <onnx_in> <onnx_out> <tensor_name> [<tensor_name> ...]
"""
import sys

import onnx
from onnx import helper, shape_inference

onnx_in, onnx_out = sys.argv[1], sys.argv[2]
names = sys.argv[3:]

model = onnx.load(onnx_in, load_external_data=False)
inferred = shape_inference.infer_shapes(model, strict_mode=False)

dtype_by_name = {vi.name: vi.type.tensor_type.elem_type for vi in inferred.graph.value_info}
for vi in list(inferred.graph.output) + list(inferred.graph.input):
    dtype_by_name[vi.name] = vi.type.tensor_type.elem_type

existing_outputs = {o.name for o in model.graph.output}
for name in names:
    if name in existing_outputs:
        continue
    if name not in dtype_by_name:
        raise RuntimeError(f"tensor {name!r} not found in graph value_info/input/output")
    model.graph.output.append(helper.make_tensor_value_info(name, dtype_by_name[name], None))
    existing_outputs.add(name)

onnx.save(model, onnx_out)
print(f"added outputs: {names}")
for name in names:
    print(f"  {name}: dtype={dtype_by_name.get(name)}")
