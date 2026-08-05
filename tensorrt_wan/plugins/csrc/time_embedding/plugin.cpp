#include "time_embedding/plugin.h"
#include "common/plugin_creator_base.h"
#include <cstring>

namespace tensorrt_wan {
namespace plugins {

const std::vector<nvinfer1::PluginField>& TimeEmbeddingPlugin::fieldTemplate() {
  static const std::vector<nvinfer1::PluginField> fields = {
      {"dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
      {"max_period", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
  };
  return fields;
}

TimeEmbeddingPlugin* TimeEmbeddingPlugin::createFromFields(const nvinfer1::PluginFieldCollection* fc) {
  int dim = 256;
  float maxPeriod = 10000.f;
  for (int i = 0; i < fc->nbFields; ++i) {
    const auto& f = fc->fields[i];
    const std::string n(f.name);
    if (n == "dim") dim = *static_cast<const int32_t*>(f.data);
    else if (n == "max_period") maxPeriod = *static_cast<const float*>(f.data);
  }
  return new TimeEmbeddingPlugin(dim, maxPeriod);
}

TimeEmbeddingPlugin* TimeEmbeddingPlugin::createFromSerialized(const void* data, size_t length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  int dim;
  float maxPeriod;
  std::memcpy(&dim, bytes, sizeof(dim));
  std::memcpy(&maxPeriod, bytes + sizeof(dim), sizeof(maxPeriod));
  return new TimeEmbeddingPlugin(dim, maxPeriod);
}

void TimeEmbeddingPlugin::serializeParams(void* buffer) const {
  auto* bytes = static_cast<uint8_t*>(buffer);
  std::memcpy(bytes, &dim_, sizeof(dim_));
  std::memcpy(bytes + sizeof(dim_), &maxPeriod_, sizeof(maxPeriod_));
}

nvinfer1::DimsExprs TimeEmbeddingPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                              int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
  nvinfer1::DimsExprs out;
  out.nbDims = 2;
  out.d[0] = inputs[0].d[0];
  out.d[1] = exprBuilder.constant(dim_);
  return out;
}

int TimeEmbeddingPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                                  const void* const* inputs, void* const* outputs, void* workspace,
                                  cudaStream_t stream) noexcept {
  const int batch = inputDesc[0].dims.d[0];
  const bool isHalf = outputDesc[0].type == nvinfer1::DataType::kHALF;
  launchTimeEmbedding(inputs[0], outputs[0], batch, dim_, maxPeriod_, isHalf, stream);
  return 0;
}

}  // namespace plugins
}  // namespace tensorrt_wan

using tensorrt_wan::plugins::PluginCreatorBase;
using tensorrt_wan::plugins::TimeEmbeddingPlugin;
using TimeEmbeddingCreator = PluginCreatorBase<TimeEmbeddingPlugin>;
REGISTER_TENSORRT_PLUGIN(TimeEmbeddingCreator);
