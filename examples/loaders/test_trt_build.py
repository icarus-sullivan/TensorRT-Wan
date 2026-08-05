import time

import tensorrt as trt

ONNX_PATH = "/workspace/runpod-slim/dit_high_noise_test.onnx"
ENGINE_PATH = "/workspace/runpod-slim/dit_high_noise_test.engine"

trt_logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(trt_logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser = trt.OnnxParser(network, trt_logger)

print("parsing ONNX...")
t0 = time.time()
success = parser.parse_from_file(ONNX_PATH)
if not success:
    for i in range(parser.num_errors):
        print("PARSE ERROR:", parser.get_error(i))
    raise SystemExit("ONNX parse failed")
print(f"parsed OK in {time.time() - t0:.1f}s — {network.num_layers} layers, "
      f"{network.num_inputs} inputs, {network.num_outputs} outputs")
for i in range(network.num_inputs):
    inp = network.get_input(i)
    print(f"  input[{i}]: {inp.name} {inp.shape} {inp.dtype}")
for i in range(network.num_outputs):
    out = network.get_output(i)
    print(f"  output[{i}]: {out.name} {out.shape} {out.dtype}")

config = builder.create_builder_config()
# No precision BuilderFlag needed (and none exist in this TensorRT version's API for it, see
# docs/wan2.2_i2v_14b_notes.md) — STRONGLY_TYPED networks take their precision entirely from the
# ONNX graph's own tensor dtypes, already fp16 throughout per the parse output above.

print("building engine (real model, 40 blocks — may take a while, let it run)...")
t0 = time.time()
serialized = builder.build_serialized_network(network, config)
elapsed = time.time() - t0
if serialized is None:
    raise SystemExit(f"engine build FAILED after {elapsed:.1f}s")

print(f"build SUCCEEDED in {elapsed:.1f}s")
# IHostMemory doesn't support len() directly in this TensorRT version's bindings - convert to
# bytes first, same as export/trt_build.py's build_tensorrt_engine already does.
serialized_bytes = bytes(serialized)
print(f"engine size: {len(serialized_bytes) / (1 << 30):.2f} GiB")

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized_bytes)
print("saved to", ENGINE_PATH)
