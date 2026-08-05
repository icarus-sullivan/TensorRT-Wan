// Adaptive LayerNorm (AdaLN-Zero, as used by DiT-family backbones including Wan).
//
// Inputs:  x (B, S, C), scale (B, C), shift (B, C)
// Output:  LayerNorm_no_affine(x) * (1 + scale) + shift, shape (B, S, C)
//
// LayerNorm here has no learned affine params (elementwise_affine=False) — scale/shift come
// entirely from the timestep/conditioning MLP upstream, per the AdaLN-Zero design. Validate
// against the PyTorch reference during RunPod op-parity testing (docs/plugins.md) before trusting
// numerics in a built engine.
#pragma once

#include "common/plugin_base.h"
#include <cuda_runtime_api.h>

namespace tensorrt_wan {
namespace plugins {

class AdaLayerNormPlugin : public PluginBase {
 public:
  static constexpr const char* kName = "AdaLayerNorm";
  static constexpr const char* kVersion = "1";

  explicit AdaLayerNormPlugin(float eps = 1e-6f) : eps_(eps) {}

  static const std::vector<nvinfer1::PluginField>& fieldTemplate();
  static AdaLayerNormPlugin* createFromFields(const nvinfer1::PluginFieldCollection* fc);
  static AdaLayerNormPlugin* createFromSerialized(const void* data, size_t length);

  int getNbOutputs() const noexcept override { return 1; }

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                           nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

 protected:
  const char* pluginType() const override { return kName; }
  const char* pluginVersion() const override { return kVersion; }
  size_t serializedSize() const override { return sizeof(eps_); }
  void serializeParams(void* buffer) const override { std::memcpy(buffer, &eps_, sizeof(eps_)); }
  PluginBase* cloneImpl() const override { return new AdaLayerNormPlugin(eps_); }

 private:
  float eps_;
};

void launchAdaLayerNorm(const void* x, const void* scale, const void* shift, void* out, int batch, int seq,
                         int channels, float eps, bool isHalf, cudaStream_t stream);

}  // namespace plugins
}  // namespace tensorrt_wan
