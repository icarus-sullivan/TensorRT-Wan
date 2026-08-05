// Rotary position embedding (RoPE), applied to attention Q/K projections.
//
// Inputs:  x (B, H, S, D), cos (S, D), sin (S, D)
// Output:  x_rotated (B, H, S, D), same dtype/shape as x
//
// Uses the standard "rotate-half" formulation (RoFormer / DiT convention): split the last
// dimension in half as [x1, x2], output = [x1*cos1 - x2*sin1, x2*cos2 + x1*sin2]. This matches
// the RoPE convention used across the DiT/Wan model family; validate numerically against the
// PyTorch reference during the RunPod op-parity phase (docs/plugins.md) before trusting it in a
// built engine — Wan-specific interleaving (rotate-half vs. rotate-every-two) is not verified
// against upstream source in this environment.
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
