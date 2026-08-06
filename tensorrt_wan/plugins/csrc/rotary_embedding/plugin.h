// Rotary position embedding (RoPE), applied to attention Q/K projections.
//
// Inputs:  x (B, H, S, D), cos (S, D), sin (S, D)
// Output:  x_rotated (B, H, S, D), same dtype/shape as x
//
// Uses Wan's actual convention: interleaved-pair rotation, confirmed against
// `comfy/ldm/flux/math.py`'s `_apply_rope1` (cloned in
// examples/loaders/wan_comfyui_loader.py) — NOT the "rotate-half" split-first-half/
// second-half convention this kernel implemented previously. Each pair of *adjacent*
// elements (x[2i], x[2i+1]) is rotated together by a single angle:
//   out[2i]   = x[2i]*cos_i - x[2i+1]*sin_i
//   out[2i+1] = x[2i]*sin_i + x[2i+1]*cos_i
// `cos`/`sin` must be pre-duplicated per pair (cos[2i] == cos[2i+1] == cos_i) — i.e.
// repeat-interleaved, not concat-duplicated (concat-duplicate is the rotate-half table
// layout). Still needs numerical validation against the PyTorch reference during the
// RunPod op-parity phase (docs/plugins.md).
#pragma once

#include "common/plugin_base.h"
#include <cuda_runtime_api.h>

namespace tensorrt_wan {
namespace plugins {

class RotaryEmbeddingPlugin : public PluginBase {
 public:
  static constexpr const char* kName = "RotaryEmbedding";
  static constexpr const char* kVersion = "1";

  explicit RotaryEmbeddingPlugin() = default;

  static const std::vector<nvinfer1::PluginField>& fieldTemplate();
  static RotaryEmbeddingPlugin* createFromFields(const nvinfer1::PluginFieldCollection* fc);
  static RotaryEmbeddingPlugin* createFromSerialized(const void* data, size_t length);

  int getNbOutputs() const noexcept override { return 1; }

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                           int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

 protected:
  const char* pluginType() const override { return kName; }
  const char* pluginVersion() const override { return kVersion; }
  size_t serializedSize() const override { return 0; }
  void serializeParams(void* buffer) const override {}
  PluginBase* cloneImpl() const override { return new RotaryEmbeddingPlugin(); }
};

// Kernel entry point, defined in kernel.cu.
void launchRotaryEmbedding(const void* x, const void* cos, const void* sin, void* out, int batch,
                            int heads, int seq, int headDim, bool isHalf, cudaStream_t stream);

}  // namespace plugins
}  // namespace tensorrt_wan
