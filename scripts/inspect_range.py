"""Ad-hoc: list op_types for a node-index range of an ONNX graph, CPU-only, no weights loaded.

Usage: python3 inspect_range.py <onnx_path> <start_idx> <end_idx> [> logfile]
"""
import sys

import onnx

path = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])

model = onnx.load(path, load_external_data=False)
nodes = model.graph.node
print(f"graph has {len(nodes)} nodes; showing [{start}:{end}]")
for i in range(start, min(end, len(nodes))):
    n = nodes[i]
    inputs = ", ".join(n.input)
    outputs = ", ".join(n.output)
    print(f"[{i}] {n.op_type}  name={n.name!r}  inputs=({inputs})  outputs=({outputs})")
