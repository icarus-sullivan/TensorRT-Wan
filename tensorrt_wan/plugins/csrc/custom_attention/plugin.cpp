#include "custom_attention/plugin.h"
#include "common/plugin_creator_base.h"
#include <cstring>

namespace tensorrt_wan {
namespace plugins {

const std::vector<nvinfer1::PluginField>& CustomAttentionPlugin::fieldTemplate() {
  static const std::vector<nvinfer1::PluginField> fields = {
      {"backend", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
      {"causal", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
  };
  return fields;
}

CustomAttentionPlugin* CustomAttentionPlugin::createFromFields(const nvinfer1::PluginFieldCollection* fc) {
  auto backend = AttentionBackend::kAuto;
  bool causal = false;
  for (int i = 0; i < fc->nbFields; ++i) {
    const auto& f = fc->fields[i];
    if (std::string(f.name) == "backend")
      backend = static_cast<AttentionBackend>(*static_cast<const int32_t*>(f.data));
    else if (std::string(f.name) == "causal")
      causal = *static_cast<const int32_t*>(f.data) != 0;
  }
  return new CustomAttentionPlugin(backend, causal);
}

CustomAttentionPlugin* CustomAttentionPlugin::createFromSerialized(const void* data, size_t length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  AttentionBackend backend;
  bool causal;
  std::memcpy(&backend, bytes, sizeof(backend));
  std::memcpy(&causal, bytes + sizeof(backend), sizeof(causal));
  return new CustomAttentionPlugin(backend, causal);
}

void CustomAttentionPlugin::serializeParams(void* buffer) const {
  auto* bytes = static_cast<uint8_t*>(buffer);
  std::memcpy(bytes, &backend_, sizeof(backend_));
  std::memcpy(bytes + sizeof(backend_), &causal_, sizeof(causal_));
}

nvinfer1::DimsExprs CustomAttentionPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                                int nbInputs,
                                                                nvinfer1::IExprBuilder& exprBuilder) noexcept {
  return inputs[0];  // (B, H, S, D), same as q
}

int CustomAttentionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  const auto& dims = inputDesc[0].dims;  // (B, H, S, D)
  const bool isHalf = inputDesc[0].type == nvinfer1::DataType::kHALF;
  dispatchAttention(backend_, causal_, inputs[0], inputs[1], inputs[2], outputs[0], dims.d[0], dims.d[1], dims.d[2],
                     dims.d[3], isHalf, stream);
  return 0;
}

}  // namespace plugins
}  // namespace tensorrt_wan

using tensorrt_wan::plugins::CustomAttentionPlugin;
using tensorrt_wan::plugins::PluginCreatorBase;
using CustomAttentionCreator = PluginCreatorBase<CustomAttentionPlugin>;
REGISTER_TENSORRT_PLUGIN(CustomAttentionCreator);
