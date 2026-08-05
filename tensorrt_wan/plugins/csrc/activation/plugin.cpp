#include "activation/plugin.h"
#include "common/plugin_creator_base.h"
#include <cstring>
#include <numeric>

namespace tensorrt_wan {
namespace plugins {

const std::vector<nvinfer1::PluginField>& ActivationPlugin::fieldTemplate() {
  static const std::vector<nvinfer1::PluginField> fields = {
      {"kind", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
  };
  return fields;
}

ActivationPlugin* ActivationPlugin::createFromFields(const nvinfer1::PluginFieldCollection* fc) {
  auto kind = ActivationKind::kSiLU;
  for (int i = 0; i < fc->nbFields; ++i) {
    if (std::string(fc->fields[i].name) == "kind")
      kind = static_cast<ActivationKind>(*static_cast<const int32_t*>(fc->fields[i].data));
  }
  return new ActivationPlugin(kind);
}

ActivationPlugin* ActivationPlugin::createFromSerialized(const void* data, size_t length) {
  ActivationKind kind;
  std::memcpy(&kind, data, sizeof(kind));
  return new ActivationPlugin(kind);
}

int ActivationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                               const void* const* inputs, void* const* outputs, void* workspace,
                               cudaStream_t stream) noexcept {
  const auto& dims = inputDesc[0].dims;
  long numel = 1;
  for (int i = 0; i < dims.nbDims; ++i) numel *= dims.d[i];
  const bool isHalf = inputDesc[0].type == nvinfer1::DataType::kHALF;
  launchActivation(kind_, inputs[0], outputs[0], numel, isHalf, stream);
  return 0;
}

}  // namespace plugins
}  // namespace tensorrt_wan

using tensorrt_wan::plugins::ActivationPlugin;
using tensorrt_wan::plugins::PluginCreatorBase;
using ActivationCreator = PluginCreatorBase<ActivationPlugin>;
REGISTER_TENSORRT_PLUGIN(ActivationCreator);
