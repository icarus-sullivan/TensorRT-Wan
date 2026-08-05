// Custom scaled-dot-product attention plugin: a TensorRT-pluggable dispatcher over
// FlashAttention / FlashAttention-2 / FlashAttention-3 / SageAttention, selected per
// `config.AttentionConfig.implementation` (see runtime/precision.py's sibling selection logic
// for attention). This plugin intentionally does NOT reimplement attention math — those kernels
// are large, hardware-specific, and already exist in well-validated external libraries; this
// plugin's job is to make whichever one is selected callable from inside a TensorRT engine.
//
// Inputs:  q, k, v (B, H, S, D)
// Output:  attn_out (B, H, S, D)
//
// `dispatchAttention()` in kernel_dispatch.cpp is the integration seam: wiring it to a real
// FlashAttention/SageAttention backend (vendoring the library, calling its C++ API) is GPU-build
// work done in the validation phase, not in this structure-only phase.
#pragma once

#include "common/plugin_base.h"
#include <cuda_runtime_api.h>
#include <string>

namespace tensorrt_wan {
namespace plugins {

enum class AttentionBackend { kAuto, kFlashAttention, kFlashAttention2, kFlashAttention3, kSageAttention };

class CustomAttentionPlugin : public PluginBase {
 public:
  static constexpr const char* kName = "CustomAttention";
  static constexpr const char* kVersion = "1";

  explicit CustomAttentionPlugin(AttentionBackend backend = AttentionBackend::kAuto, bool causal = false)
      : backend_(backend), causal_(causal) {}

  static const std::vector<nvinfer1::PluginField>& fieldTemplate();
  static CustomAttentionPlugin* createFromFields(const nvinfer1::PluginFieldCollection* fc);
  static CustomAttentionPlugin* createFromSerialized(const void* data, size_t length);

  int getNbOutputs() const noexcept override { return 1; }

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                           nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

 protected:
  const char* pluginType() const override { return kName; }
  const char* pluginVersion() const override { return kVersion; }
  size_t serializedSize() const override { return sizeof(backend_) + sizeof(causal_); }
  void serializeParams(void* buffer) const override;
  PluginBase* cloneImpl() const override { return new CustomAttentionPlugin(backend_, causal_); }

 private:
  AttentionBackend backend_;
  bool causal_;
};

// Selects and invokes the configured attention backend. See file header — not yet wired to a
// real FlashAttention/SageAttention implementation in this phase.
void dispatchAttention(AttentionBackend backend, bool causal, const void* q, const void* k, const void* v,
                        void* out, int batch, int heads, int seq, int headDim, bool isHalf, cudaStream_t stream);

}  // namespace plugins
}  // namespace tensorrt_wan
