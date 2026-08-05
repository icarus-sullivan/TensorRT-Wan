#include "adalayernorm/plugin.h"
#include "common/plugin_creator_base.h"

namespace tensorrt_wan {
namespace plugins {

const std::vector<nvinfer1::PluginField>& AdaLayerNormPlugin::fieldTemplate() {
  static const std::vector<nvinfer1::PluginField> fields = {
      {"eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
  };
  return fields;
}

AdaLayerNormPlugin* AdaLayerNormPlugin::createFromFields(const nvinfer1::PluginFieldCollection* fc) {
  float eps = 1e-6f;
  for (int i = 0; i < fc->nbFields; ++i) {
    const auto& f = fc->fields[i];
    if (std::string(f.name) == "eps") eps = *static_cast<const float*>(f.data);
  }
  return new AdaLayerNormPlugin(eps);
}

AdaLayerNormPlugin* AdaLayerNormPlugin::createFromSerialized(const void* data, size_t length) {
  float eps;
  std::memcpy(&eps, data, sizeof(eps));
  return new AdaLayerNormPlugin(eps);
}

nvinfer1::DimsExprs AdaLayerNormPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                             int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
  return inputs[0];
}

int AdaLayerNormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                                 const void* const* inputs, void* const* outputs, void* workspace,
                                 cudaStream_t stream) noexcept {
  const auto& dims = inputDesc[0].dims;  // (B, S, C)
  const bool isHalf = inputDesc[0].type == nvinfer1::DataType::kHALF;
  launchAdaLayerNorm(inputs[0], inputs[1], inputs[2], outputs[0], dims.d[0], dims.d[1], dims.d[2], eps_, isHalf,
                      stream);
  return 0;
}

}  // namespace plugins
}  // namespace tensorrt_wan

using tensorrt_wan::plugins::AdaLayerNormPlugin;
using tensorrt_wan::plugins::PluginCreatorBase;
using AdaLayerNormCreator = PluginCreatorBase<AdaLayerNormPlugin>;
REGISTER_TENSORRT_PLUGIN(AdaLayerNormCreator);
