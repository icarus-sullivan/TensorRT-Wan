// Sinusoidal timestep embedding (standard DDPM/diffusion convention).
//
// Inputs:  timestep (B,)
// Output:  embedding (B, dim) = concat(cos(t * freqs), sin(t * freqs)), freqs geometrically
//          spaced over dim/2 frequencies. Feeds into the (external, ordinary Linear-layer) time
//          MLP — that MLP is left as regular ONNX ops, only the sinusoidal projection itself
//          needs a plugin since ONNX/TensorRT has no native "diffusion timestep embedding" op.
#pragma once

#include "common/plugin_base.h"
#include <cuda_runtime_api.h>

namespace tensorrt_wan {
namespace plugins {

class TimeEmbeddingPlugin : public PluginBase {
 public:
  static constexpr const char* kName = "TimeEmbedding";
  static constexpr const char* kVersion = "1";

  TimeEmbeddingPlugin(int dim, float maxPeriod = 10000.f) : dim_(dim), maxPeriod_(maxPeriod) {}

  static const std::vector<nvinfer1::PluginField>& fieldTemplate();
  static TimeEmbeddingPlugin* createFromFields(const nvinfer1::PluginFieldCollection* fc);
  static TimeEmbeddingPlugin* createFromSerialized(const void* data, size_t length);

  int getNbOutputs() const noexcept override { return 1; }
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                           nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

 protected:
  const char* pluginType() const override { return kName; }
  const char* pluginVersion() const override { return kVersion; }
  size_t serializedSize() const override { return sizeof(dim_) + sizeof(maxPeriod_); }
  void serializeParams(void* buffer) const override;
  PluginBase* cloneImpl() const override { return new TimeEmbeddingPlugin(dim_, maxPeriod_); }

 private:
  int dim_;
  float maxPeriod_;
};

void launchTimeEmbedding(const void* timestep, void* out, int batch, int dim, float maxPeriod, bool isHalf,
                          cudaStream_t stream);

}  // namespace plugins
}  // namespace tensorrt_wan
