// Patch embedding: 3D-conv-equivalent patchify + linear projection of a latent video tensor
// into DiT input tokens (stride == kernel size, i.e. non-overlapping patches).
//
// Inputs:  latents (B, C, T, H, W), weight (embedDim, C*pt*ph*pw), bias (embedDim)
// Output:  tokens (B, (T/pt)*(H/ph)*(W/pw), embedDim)
#pragma once

#include "common/plugin_base.h"
#include <cuda_runtime_api.h>

namespace tensorrt_wan {
namespace plugins {

class PatchEmbedPlugin : public PluginBase {
 public:
  static constexpr const char* kName = "PatchEmbed";
  static constexpr const char* kVersion = "1";

  PatchEmbedPlugin(int patchT, int patchH, int patchW) : patchT_(patchT), patchH_(patchH), patchW_(patchW) {}

  static const std::vector<nvinfer1::PluginField>& fieldTemplate();
  static PatchEmbedPlugin* createFromFields(const nvinfer1::PluginFieldCollection* fc);
  static PatchEmbedPlugin* createFromSerialized(const void* data, size_t length);

  int getNbOutputs() const noexcept override { return 1; }
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                           nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

 protected:
  const char* pluginType() const override { return kName; }
  const char* pluginVersion() const override { return kVersion; }
  size_t serializedSize() const override { return 3 * sizeof(int); }
  void serializeParams(void* buffer) const override;
  PluginBase* cloneImpl() const override { return new PatchEmbedPlugin(patchT_, patchH_, patchW_); }

 private:
  int patchT_, patchH_, patchW_;
};

void launchPatchEmbed(const void* latents, const void* weight, const void* bias, void* out, int batch, int channels,
                       int t, int h, int w, int patchT, int patchH, int patchW, int embedDim, bool isHalf,
                       cudaStream_t stream);

}  // namespace plugins
}  // namespace tensorrt_wan
