#include "rotary_embedding/plugin.h"
#include "common/plugin_creator_base.h"

namespace tensorrt_wan {
namespace plugins {

const std::vector<nvinfer1::PluginField>& RotaryEmbeddingPlugin::fieldTemplate() {
  static const std::vector<nvinfer1::PluginField> fields;  // no build-time params; shapes are runtime
  return fields;
}

RotaryEmbeddingPlugin* RotaryEmbeddingPlugin::createFromFields(const nvinfer1::PluginFieldCollection* fc) {
  return new RotaryEmbeddingPlugin();
}

RotaryEmbeddingPlugin* RotaryEmbeddingPlugin::createFromSerialized(const void* data, size_t length) {
  return new RotaryEmbeddingPlugin();
}

nvinfer1::DimsExprs RotaryEmbeddingPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                                int nbInputs,
                                                                nvinfer1::IExprBuilder& exprBuilder) noexcept {
  return inputs[0];  // output shape == input "x" shape
}

int RotaryEmbeddingPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  const auto& dims = inputDesc[0].dims;  // (B, H, S, D)
  const int batch = dims.d[0];
  const int heads = dims.d[1];
  const int seq = dims.d[2];
  const int headDim = dims.d[3];
  const bool isHalf = inputDesc[0].type == nvinfer1::DataType::kHALF;

  launchRotaryEmbedding(inputs[0], inputs[1], inputs[2], outputs[0], batch, heads, seq, headDim, isHalf, stream);
  return 0;
}

}  // namespace plugins
}  // namespace tensorrt_wan

using tensorrt_wan::plugins::PluginCreatorBase;
using tensorrt_wan::plugins::RotaryEmbeddingPlugin;
using RotaryEmbeddingCreator = PluginCreatorBase<RotaryEmbeddingPlugin>;
REGISTER_TENSORRT_PLUGIN(RotaryEmbeddingCreator);
