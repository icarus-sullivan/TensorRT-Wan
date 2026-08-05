#include "patch_reconstruct/plugin.h"
#include "common/plugin_creator_base.h"
#include <cstring>

namespace tensorrt_wan {
namespace plugins {

const std::vector<nvinfer1::PluginField>& PatchReconstructPlugin::fieldTemplate() {
  static const std::vector<nvinfer1::PluginField> fields = {
      {"channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
      {"patch_t", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
      {"patch_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
      {"patch_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
      {"out_h_patches", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
      {"out_w_patches", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
  };
  return fields;
}

PatchReconstructPlugin* PatchReconstructPlugin::createFromFields(const nvinfer1::PluginFieldCollection* fc) {
  int c = 16, pt = 1, ph = 2, pw = 2, nh = 0, nw = 0;
  for (int i = 0; i < fc->nbFields; ++i) {
    const auto& f = fc->fields[i];
    const std::string n(f.name);
    const int v = *static_cast<const int32_t*>(f.data);
    if (n == "channels") c = v;
    else if (n == "patch_t") pt = v;
    else if (n == "patch_h") ph = v;
    else if (n == "patch_w") pw = v;
    else if (n == "out_h_patches") nh = v;
    else if (n == "out_w_patches") nw = v;
  }
  return new PatchReconstructPlugin(c, pt, ph, pw, nh, nw);
}

PatchReconstructPlugin* PatchReconstructPlugin::createFromSerialized(const void* data, size_t length) {
  int vals[6];
  std::memcpy(vals, data, sizeof(vals));
  return new PatchReconstructPlugin(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
}

void PatchReconstructPlugin::serializeParams(void* buffer) const {
  int vals[6] = {channels_, patchT_, patchH_, patchW_, outH_, outW_};
  std::memcpy(buffer, vals, sizeof(vals));
}

nvinfer1::DimsExprs PatchReconstructPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                                 int nbInputs,
                                                                 nvinfer1::IExprBuilder& exprBuilder) noexcept {
  // inputs[0] = tokens (B, N, embedDim). N = nT * outH_ * outW_ (outH_/outW_ given in patch units).
  nvinfer1::DimsExprs out;
  out.nbDims = 5;
  out.d[0] = inputs[0].d[0];
  out.d[1] = exprBuilder.constant(channels_);
  auto* hw = exprBuilder.constant(outH_ * outW_);
  auto* nT = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *inputs[0].d[1], *hw);
  out.d[2] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *nT, *exprBuilder.constant(patchT_));
  out.d[3] = exprBuilder.constant(outH_ * patchH_);
  out.d[4] = exprBuilder.constant(outW_ * patchW_);
  return out;
}

int PatchReconstructPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                     const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                     void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  const auto& dims = inputDesc[0].dims;  // (B, N, embedDim)
  const bool isHalf = inputDesc[0].type == nvinfer1::DataType::kHALF;
  launchPatchReconstruct(inputs[0], inputs[1], inputs[2], outputs[0], dims.d[0], dims.d[1], dims.d[2], channels_,
                          patchT_, patchH_, patchW_, outH_, outW_, isHalf, stream);
  return 0;
}

}  // namespace plugins
}  // namespace tensorrt_wan

using tensorrt_wan::plugins::PatchReconstructPlugin;
using tensorrt_wan::plugins::PluginCreatorBase;
using PatchReconstructCreator = PluginCreatorBase<PatchReconstructPlugin>;
REGISTER_TENSORRT_PLUGIN(PatchReconstructCreator);
