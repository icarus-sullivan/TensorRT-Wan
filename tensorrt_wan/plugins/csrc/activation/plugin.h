// Fused activation plugin: SiLU / GELU (tanh approx) / QuickGELU, selected at build time.
//
// Most engines can rely on TensorRT's native activation layers; this plugin exists for the cases
// where fusing the activation into a neighboring custom plugin (e.g. a future fused MLP) avoids a
// kernel launch, and for QuickGELU (x * sigmoid(1.702x)), which TensorRT has no native op for.
//
// Input/Output: x, same shape, elementwise.
#pragma once

#include "common/plugin_base.h"
#include <cuda_runtime_api.h>

namespace tensorrt_wan {
namespace plugins {

enum class ActivationKind { kSiLU = 0, kGELU = 1, kQuickGELU = 2 };

class ActivationPlugin : public PluginBase {
 public:
  static constexpr const char* kName = "FusedActivation";
  static constexpr const char* kVersion = "1";

  explicit ActivationPlugin(ActivationKind kind) : kind_(kind) {}

  static const std::vector<nvinfer1::PluginField>& fieldTemplate();
  static ActivationPlugin* createFromFields(const nvinfer1::PluginFieldCollection* fc);
  static ActivationPlugin* createFromSerialized(const void* data, size_t length);

  int getNbOutputs() const noexcept override { return 1; }
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                           nvinfer1::IExprBuilder& exprBuilder) noexcept override {
    return inputs[0];
  }
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

 protected:
  const char* pluginType() const override { return kName; }
  const char* pluginVersion() const override { return kVersion; }
  size_t serializedSize() const override { return sizeof(kind_); }
  void serializeParams(void* buffer) const override { std::memcpy(buffer, &kind_, sizeof(kind_)); }
  PluginBase* cloneImpl() const override { return new ActivationPlugin(kind_); }

 private:
  ActivationKind kind_;
};

void launchActivation(ActivationKind kind, const void* x, void* out, long numel, bool isHalf, cudaStream_t stream);

}  // namespace plugins
}  // namespace tensorrt_wan
