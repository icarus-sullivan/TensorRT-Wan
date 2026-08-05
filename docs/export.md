# Export Process (PyTorch -> ONNX)

## Why TensorRT-Wan doesn't vendor Wan's model code

Wan releases new architectures independently of this framework, and this project is meant to
track future Wan/TensorRT/CUDA releases (PLAN.md). Hardcoding a model loader here would need
updating every release. Instead, export/build commands take a `--loader module.path:function`
string; that function is your (or an adapter package's) responsibility and just needs to return a
loaded `nn.Module` given a checkpoint path. See
[`tensorrt_wan/cli/loader.py`](../tensorrt_wan/cli/loader.py).

## The three stages

1. **`torch.export`** ([`export/torch_export.py`](../tensorrt_wan/export/torch_export.py)) —
   traces the model with `torch.export.export`, using dynamic shape ranges declared by the
   component's `ModelExporter.dynamic_axes()`.
2. **ONNX** ([`export/onnx_export.py`](../tensorrt_wan/export/onnx_export.py)) — converts the
   `ExportedProgram` via `torch.onnx.export(..., dynamo=True)`, preserving the dynamic shapes from
   stage 1 rather than re-deriving them.
3. **TensorRT** ([`export/trt_build.py`](../tensorrt_wan/export/trt_build.py)) — parses the ONNX
   graph and builds a strongly-typed engine with one optimization profile per configured
   resolution (see [engine_generation.md](engine_generation.md)).

`export/pipeline.py`'s `run_export_pipeline()` runs all three and stores the result in the engine
cache, keyed by model/TensorRT/CUDA/GPU-arch/profile/precision (see
[`runtime/cache.py`](../tensorrt_wan/runtime/cache.py)).

## Describing a component: `ModelExporter`

Each exportable Wan submodule gets a small subclass of
[`ModelExporter`](../tensorrt_wan/export/base.py):

```python
class MyDiTExporter(ModelExporter):
    name = "dit"

    def example_inputs(self) -> dict[str, torch.Tensor]: ...
    def dynamic_axes(self) -> dict[str, list[DynamicAxis]]: ...
    input_names: list[str]
    output_names: list[str]
```

See [`export/exporters/dit.py`](../tensorrt_wan/export/exporters/dit.py),
[`text_encoder.py`](../tensorrt_wan/export/exporters/text_encoder.py), and
[`vae.py`](../tensorrt_wan/export/exporters/vae.py) for the four components this project ships
exporters for out of the box. Dimensions (`in_channels`/`latent_channels`, `text_dim`, ...) are
constructor arguments, not hardcoded constants — read them from your loaded checkpoint's own
config so a new Wan release with different dimensions needs no exporter code changes.

`DiTExporter`'s `in_channels` is the DiT's actual input channel count, not the VAE's latent
channel count — confirmed against a real Wan 2.2 14B I2V checkpoint that Wan channel-concatenates
noise-latent + image-latent + mask (16+16+4=36) before patch embedding, so `in_channels` != the
VAE's own 16-channel latent space for I2V-capable checkpoints. See
[wan2.2_i2v_14b_notes.md](wan2.2_i2v_14b_notes.md).

## CLI usage

```bash
trtwan export onnx \
  --component dit \
  --loader my_wan_adapter:load_dit \
  --checkpoint /path/to/wan2.1-t2v-14b \
  --output dit.onnx \
  --exporter-kwargs '{"in_channels": 36, "text_dim": 4096}'
```

Then [build the engine](engine_generation.md) from the resulting ONNX file.

## What this repository does NOT do (yet)

Per PLAN.md's development rule, no export has been run in this repository — only the exporters
and pipeline exist. Running the above against a real Wan checkpoint, and validating the resulting
ONNX/engine numerically against the PyTorch reference, happens on RunPod GPU hardware (see
[roadmap.md](roadmap.md)).
