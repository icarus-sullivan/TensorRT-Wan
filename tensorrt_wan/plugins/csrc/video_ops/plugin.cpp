#include "video_ops/plugin.h"
#include "common/plugin_creator_base.h"
#include <cstring>

namespace tensorrt_wan {
namespace plugins {

const std::vector<nvinfer1::PluginField>& TemporalResizePlugin::fieldTemplate() {
  static const std::vector<nvinfer1::PluginField> fields = {
      {"factor", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
  };
  return fields;
}

TemporalResizePlugin* TemporalResizePlugin::createFromFields(const nvinfer1::PluginFieldCollection* fc) {
  int factor = 2;
  for (int i = 0; i < fc->nbFields; ++i) {
    if (std::string(fc->fields[i].name) == "factor") factor = *static_cast<const int32_t*>(fc->fields[i].data);
  }
  return new TemporalResizePlugin(factor);
}

TemporalResizePlugin* TemporalResizePlugin::createFromSerialized(const void* data, size_t length) {
  int factor;
  std::memcpy(&factor, data, sizeof(factor));
  return new TemporalResizePlugin(factor);
}

nvinfer1::DimsExprs TemporalResizePlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                               int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
  nvinfer1::DimsExprs out = inputs[0];
  out.d[2] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *inputs[0].d[2], *exprBuilder.constant(factor_));
  return out;
}

int TemporalResizePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                                   const void* const* inputs, void* const* outputs, void* workspace,
                                   cudaStream_t stream) noexcept {
  const auto& dims = inputDesc[0].dims;  // (B, C, T, H, W)
  const bool isHalf = inputDesc[0].type == nvinfer1::DataType::kHALF;
  launchTemporalResize(inputs[0], outputs[0], dims.d[0], dims.d[1], dims.d[2], dims.d[3], dims.d[4], factor_, isHalf,
                        stream);
  return 0;
}

}  // namespace plugins
}  // namespace tensorrt_wan

using tensorrt_wan::plugins::PluginCreatorBase;
using tensorrt_wan::plugins::TemporalResizePlugin;
using TemporalResizeCreator = PluginCreatorBase<TemporalResizePlugin>;
REGISTER_TENSORRT_PLUGIN(TemporalResizeCreator);
