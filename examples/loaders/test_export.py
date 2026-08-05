import os
import sys
import torch

sys.path.insert(0, '/workspace/runpod-slim/TensorRT-Wan/examples/loaders')
from wan_comfyui_loader import load_dit, _DTYPES

dtype = _DTYPES[os.environ.get('TRTWAN_LOADER_DTYPE', 'fp16')]

m = load_dit('/workspace/runpod-slim/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors')
print('patch_embedding weight dtype:', m.patch_embedding.weight.dtype)

B, C, T, H, W = 1, 36, 3, 8, 8
x = torch.zeros(B, C, T, H, W, device='cuda', dtype=dtype)
timestep = torch.zeros(B, device='cuda', dtype=dtype)
context = torch.zeros(B, 32, 4096, device='cuda', dtype=dtype)
kwargs = {'x': x, 'timestep': timestep, 'context': context}

print(f'testing with dtype={dtype}')
print('running eager forward first (sanity check before export)...')
with torch.no_grad():
    out = m.forward(**kwargs)
print('eager forward OK, output shape:', out.shape)

print('attempting torch.export...')
exported = torch.export.export(m, args=(), kwargs=kwargs)
print('torch.export SUCCEEDED')

print('attempting torch.onnx.export (dynamo=True, opset=23)...')
try:
    onnx_program = torch.onnx.export(
        exported,
        (),
        kwargs=kwargs,
        input_names=['x', 'timestep', 'context'],
        output_names=['noise_pred'],
        opset_version=23,
        dynamo=True,
    )
    print('torch.onnx.export SUCCEEDED')
    out_path = '/workspace/runpod-slim/dit_high_noise_test.onnx'
    onnx_program.save(out_path)
    print('saved to', out_path)
    size_mb = os.path.getsize(out_path) / (1 << 20)
    print(f'onnx file size: {size_mb:.1f} MiB')
except Exception as e:
    print('torch.onnx.export FAILED:', type(e).__name__, str(e)[:3000])
