#include "patch_embed/plugin.h"
#include "common/plugin_creator_base.h"
#include <cstring>

namespace tensorrt_wan {
namespace plugins {

const std::vector<nvinfer1::PluginField>& PatchEmbedPlugin::fieldTemplate() {
  static const std::vector<nvinfer1::PluginField> fields = {
      {"patch_t", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
      {"patch_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
      {"patch_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
  };
  return fields;
}

PatchEmbedPlugin* PatchEmbedPlugin::createFromFields(const nvinfer1::PluginFieldCollection* fc) {
  int pt = 1, ph = 2, pw = 2;
  for (int i = 0; i < fc->nbFields; ++i) {
    const auto& f = fc->fields[i];
    const std::string n(f.name);
    const int v = *static_cast<const int32_t*>(f.data);
    if (n == "patch_t") pt = v;
    else if (n == "patch_h") ph = v;
    else if (n == "patch_w") pw = v;
  }
  return new PatchEmbedPlugin(pt, ph, pw);
}

PatchEmbedPlugin* PatchEmbedPlugin::createFromSerialized(const void* data, size_t length) {
  int vals[3];
  std::memcpy(vals, data, sizeof(vals));
  return new PatchEmbedPlugin(vals[0], vals[1], vals[2]);
}

void PatchEmbedPlugin::serializeParams(void* buffer) const {
  int vals[3] = {patchT_, patchH_, patchW_};
  std::memcpy(buffer, vals, sizeof(vals));
}

nvinfer1::DimsExprs PatchEmbedPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                           int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
  // inputs[0] = latents (B, C, T, H, W); inputs[1] = weight (embedDim, C*pt*ph*pw)
  nvinfer1::DimsExprs out;
  out.nbDims = 3;
  out.d[0] = inputs[0].d[0];  // batch
  auto* tDiv = exprBuilder.constant(patchT_);
  auto* hDiv = exprBuilder.constant(patchH_);
  auto* wDiv = exprBuilder.constant(patchW_);
  auto* nT = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *inputs[0].d[2], *tDiv);
  auto* nH = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *inputs[0].d[3], *hDiv);
  auto* nW = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *inputs[0].d[4], *wDiv);
  auto* nTH = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *nT, *nH);
  out.d[1] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *nTH, *nW);
  out.d[2] = inputs[1].d[0];  // embedDim
  return out;
}

int PatchEmbedPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                               const void* const* inputs, void* const* outputs, void* workspace,
                               cudaStream_t stream) noexcept {
  const auto& dims = inputDesc[0].dims;  // (B, C, T, H, W)
  const int embedDim = inputDesc[1].dims.d[0];
  const bool isHalf = inputDesc[0].type == nvinfer1::DataType::kHALF;
  launchPatchEmbed(inputs[0], inputs[1], inputs[2], outputs[0], dims.d[0], dims.d[1], dims.d[2], dims.d[3],
                    dims.d[4], patchT_, patchH_, patchW_, embedDim, isHalf, stream);
  return 0;
}

}  // namespace plugins
}  // namespace tensorrt_wan

using tensorrt_wan::plugins::PatchEmbedPlugin;
using tensorrt_wan::plugins::PluginCreatorBase;
using PatchEmbedCreator = PluginCreatorBase<PatchEmbedPlugin>;
REGISTER_TENSORRT_PLUGIN(PatchEmbedCreator);
