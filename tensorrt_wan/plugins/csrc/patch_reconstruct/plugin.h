// Patch reconstruction: inverse of PatchEmbed. Projects DiT output tokens back to a latent
// video grid (unpatchify), the mirror image of ../patch_embed.
//
// Inputs:  tokens (B, N, embedDim), weight (patchSize=C*pt*ph*pw, embedDim), bias (patchSize)
// Output:  latents (B, C, T, H, W) where N == (T/pt)*(H/ph)*(W/pw)
#pragma once

#include "common/plugin_base.h"
#include <cuda_runtime_api.h>

namespace tensorrt_wan {
namespace plugins {

class PatchReconstructPlugin : public PluginBase {
 public:
  static constexpr const char* kName = "PatchReconstruct";
  static constexpr const char* kVersion = "1";

  PatchReconstructPlugin(int channels, int patchT, int patchH, int patchW, int outH, int outW)
      : channels_(channels), patchT_(patchT), patchH_(patchH), patchW_(patchW), outH_(outH), outW_(outW) {}

  static const std::vector<nvinfer1::PluginField>& fieldTemplate();
  static PatchReconstructPlugin* createFromFields(const nvinfer1::PluginFieldCollection* fc);
  static PatchReconstructPlugin* createFromSerialized(const void* data, size_t length);

  int getNbOutputs() const noexcept override { return 1; }
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                           nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

 protected:
  const char* pluginType() const override { return kName; }
  const char* pluginVersion() const override { return kVersion; }
  size_t serializedSize() const override { return 6 * sizeof(int); }
  void serializeParams(void* buffer) const override;
  PluginBase* cloneImpl() const override {
    return new PatchReconstructPlugin(channels_, patchT_, patchH_, patchW_, outH_, outW_);
  }

 private:
  int channels_, patchT_, patchH_, patchW_, outH_, outW_;  // outH_/outW_ are in *patch* units (H/ph, W/pw)
};

void launchPatchReconstruct(const void* tokens, const void* weight, const void* bias, void* out, int batch,
                             int numTokens, int embedDim, int channels, int patchT, int patchH, int patchW, int nH,
                             int nW, bool isHalf, cudaStream_t stream);

}  // namespace plugins
}  // namespace tensorrt_wan
