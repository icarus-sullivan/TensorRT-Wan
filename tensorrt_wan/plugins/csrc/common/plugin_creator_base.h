// Shared IPluginCreator boilerplate, parameterized over a concrete plugin type.
//
// A plugin type `P` using this must expose:
//   static constexpr const char* kName;      // must match ../plugins/registry.py's PLUGIN_NAMES
//   static constexpr const char* kVersion;   // "1"
//   static const std::vector<nvinfer1::PluginField>& fieldTemplate();
//   static P* createFromFields(const nvinfer1::PluginFieldCollection* fc);
//   static P* createFromSerialized(const void* data, size_t length);
#pragma once

#include <NvInfer.h>
#include <NvInferRuntimePlugin.h>
#include <string>
#include <vector>

namespace tensorrt_wan {
namespace plugins {

template <typename P>
class PluginCreatorBase : public nvinfer1::IPluginCreator {
 public:
  PluginCreatorBase() {
    fields_ = P::fieldTemplate();
    fc_.nbFields = static_cast<int>(fields_.size());
    fc_.fields = fields_.data();
  }

  const char* getPluginName() const noexcept override { return P::kName; }
  const char* getPluginVersion() const noexcept override { return P::kVersion; }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fc_; }

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                     const nvinfer1::PluginFieldCollection* fc) noexcept override {
    auto* plugin = P::createFromFields(fc);
    if (plugin) plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData,
                                          size_t serialLength) noexcept override {
    auto* plugin = P::createFromSerialized(serialData, serialLength);
    if (plugin) plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  void setPluginNamespace(const char* ns) noexcept override { namespace_ = ns; }
  const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

 private:
  std::string namespace_{"tensorrt_wan"};
  std::vector<nvinfer1::PluginField> fields_;
  nvinfer1::PluginFieldCollection fc_{};
};

}  // namespace plugins
}  // namespace tensorrt_wan
