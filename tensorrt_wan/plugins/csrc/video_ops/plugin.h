// Temporal nearest-neighbor resize for video latents/tensors — used by the VAE decoder's causal
// temporal upsampling stages, which ONNX has no native "resize along an arbitrary middle axis by
// an integer factor" op for in this exact (B, C, T, H, W) layout.
//
// Inputs:  x (B, C, T, H, W)
// Output:  y (B, C, T*factor, H, W)
#pragma once

#include "common/plugin_base.h"
#include <cuda_runtime_api.h>

namespace tensorrt_wan {
namespace plugins {

class TemporalResizePlugin : public PluginBase {
 public:
  static constexpr const char* kName = "TemporalResize";
  static constexpr const char* kVersion = "1";

  explicit TemporalResizePlugin(int factor) : factor_(factor) {}

  static const std::vector<nvinfer1::PluginField>& fieldTemplate();
  static TemporalResizePlugin* createFromFields(const nvinfer1::PluginFieldCollection* fc);
  static TemporalResizePlugin* createFromSerialized(const void* data, size_t length);

  int getNbOutputs() const noexcept override { return 1; }
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                           nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

 protected:
  const char* pluginType() const override { return kName; }
  const char* pluginVersion() const override { return kVersion; }
  size_t serializedSize() const override { return sizeof(factor_); }
  void serializeParams(void* buffer) const override { std::memcpy(buffer, &factor_, sizeof(factor_)); }
  PluginBase* cloneImpl() const override { return new TemporalResizePlugin(factor_); }

 private:
  int factor_;
};

void launchTemporalResize(const void* x, void* out, int batch, int channels, int t, int h, int w, int factor,
                           bool isHalf, cudaStream_t stream);

}  // namespace plugins
}  // namespace tensorrt_wan
