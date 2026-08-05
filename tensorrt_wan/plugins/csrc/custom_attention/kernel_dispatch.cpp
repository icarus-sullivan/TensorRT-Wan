// Backend dispatch for CustomAttentionPlugin.
//
// Deliberately unimplemented in this structure-only phase: wiring this to a real
// FlashAttention-2/3 or SageAttention backend means vendoring that library and linking against
// its CUDA kernels, which is GPU-build work for the RunPod validation phase (see
// docs/plugins.md and docs/roadmap.md), not something to fake here. Calling this now fails
// loudly via runtime.fallback rather than silently producing wrong attention output.
#include "custom_attention/plugin.h"
#include <stdexcept>

namespace tensorrt_wan {
namespace plugins {

void dispatchAttention(AttentionBackend backend, bool causal, const void* q, const void* k, const void* v,
                        void* out, int batch, int heads, int seq, int headDim, bool isHalf, cudaStream_t stream) {
  throw std::runtime_error(
      "CustomAttentionPlugin::dispatchAttention has no backend wired up yet; "
      "see docs/plugins.md for the FlashAttention/SageAttention integration status.");
}

}  // namespace plugins
}  // namespace tensorrt_wan
