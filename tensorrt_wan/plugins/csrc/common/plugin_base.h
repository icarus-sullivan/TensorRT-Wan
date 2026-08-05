// Shared IPluginV2DynamicExt boilerplate for every TensorRT-Wan plugin.
//
// Every op in ../<op>/plugin.h extends PluginBase and only implements what actually varies:
// getOutputDimensions(), enqueue(), and (de)serialization of its own parameters. Everything
// TensorRT requires but that is identical across every plugin here (clone/destroy bookkeeping,
// FP16/FP32 format support, workspace sizing, the plugin-creator field-collection dance) lives
// in this one file so a TensorRT API version bump is a one-file fix, not an eight-file one.
//
// Not compiled or linked in this development phase — see docs/plugins.md. Built and validated
// against the reference PyTorch ops on RunPod GPU hardware in the validation phase.
#pragma once

#include <NvInfer.h>
#include <cstring>
#include <string>
#include <vector>

namespace tensorrt_wan {
namespace plugins {

// One (name, value) pair a plugin creator exposes via getFieldNames(), and a plugin instance
// receives via its constructor. Kept simple (float/int/string) since none of the ops in this
// project need anything richer.
struct PluginField {
  std::string name;
  enum class Type { kFLOAT, kINT32, kSTRING } type;
};

// Base for every op's IPluginV2DynamicExt implementation.
//
// Subclasses must implement:
//   const char* pluginType() const
//   const char* pluginVersion() const
//   nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
//                                           int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
//   int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
//               const nvinfer1::PluginTensorDesc* outputDesc,
//               const void* const* inputs, void* const* outputs,
//               void* workspace, cudaStream_t stream)
//   size_t serializedSize() const
//   void serializeParams(void* buffer) const
//   PluginBase* cloneImpl() const
class PluginBase : public nvinfer1::IPluginV2DynamicExt {
 public:
  ~PluginBase() override = default;

  // --- Format support: FP32 and FP16 linear layout on every I/O tensor, uniformly. Ops that
  // need FP8/BF16 support (see runtime.precision) override this once they have a validated
  // kernel for that type.
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                  int nbOutputs) noexcept override {
    const auto& desc = inOut[pos];
    return (desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF) &&
           desc.format == nvinfer1::TensorFormat::kLINEAR;
  }

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                        int nbInputs) const noexcept override {
    return inputTypes[0];
  }

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                           const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override {
    return 0;
  }

  int initialize() noexcept override { return 0; }
  void terminate() noexcept override {}
  void destroy() noexcept override { delete this; }

  void setPluginNamespace(const char* ns) noexcept override { namespace_ = ns; }
  const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  // Subclasses report their own (name, version) via pluginType()/pluginVersion(); these forward
  // to those so IPluginV2's flat API and this project's per-op naming stay in one place.
  const char* getPluginType() const noexcept override { return pluginType(); }
  const char* getPluginVersion() const noexcept override { return pluginVersion(); }

  size_t getSerializationSize() const noexcept override { return serializedSize(); }
  void serialize(void* buffer) const noexcept override { serializeParams(buffer); }

  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override { return cloneImpl(); }

 protected:
  virtual const char* pluginType() const = 0;
  virtual const char* pluginVersion() const = 0;
  virtual size_t serializedSize() const = 0;
  virtual void serializeParams(void* buffer) const = 0;
  virtual PluginBase* cloneImpl() const = 0;

  std::string namespace_;
};

}  // namespace plugins
}  // namespace tensorrt_wan
