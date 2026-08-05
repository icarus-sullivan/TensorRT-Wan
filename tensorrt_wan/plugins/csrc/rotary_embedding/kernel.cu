#include "rotary_embedding/plugin.h"
#include <cuda_fp16.h>

namespace tensorrt_wan {
namespace plugins {

// One thread per (b, h, s, d) with d in [0, headDim/2) — each thread produces both halves of the
// rotated pair, matching the "rotate-half" split described in plugin.h.
template <typename T>
__global__ void rotaryEmbeddingKernel(const T* __restrict__ x, const T* __restrict__ cosTab,
                                       const T* __restrict__ sinTab, T* __restrict__ out, int heads, int seq,
                                       int headDim) {
  const int half = headDim / 2;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= half) return;

  const int s = blockIdx.y;
  const int bh = blockIdx.z;  // flattened (batch, head) index
  const int h = bh % heads;

  const long base = (long)bh * seq * headDim + (long)s * headDim;
  const long cosSinBase = (long)s * headDim;

  const float x1 = static_cast<float>(x[base + d]);
  const float x2 = static_cast<float>(x[base + half + d]);
  const float c1 = static_cast<float>(cosTab[cosSinBase + d]);
  const float s1 = static_cast<float>(sinTab[cosSinBase + d]);
  const float c2 = static_cast<float>(cosTab[cosSinBase + half + d]);
  const float s2 = static_cast<float>(sinTab[cosSinBase + half + d]);

  out[base + d] = static_cast<T>(x1 * c1 - x2 * s1);
  out[base + half + d] = static_cast<T>(x2 * c2 + x1 * s2);
}

void launchRotaryEmbedding(const void* x, const void* cosTab, const void* sinTab, void* out, int batch, int heads,
                            int seq, int headDim, bool isHalf, cudaStream_t stream) {
  const int half = headDim / 2;
  const dim3 block(std::min(half, 256));
  const dim3 grid((half + block.x - 1) / block.x, seq, batch * heads);

  if (isHalf) {
    rotaryEmbeddingKernel<__half><<<grid, block, 0, stream>>>(
        static_cast<const __half*>(x), static_cast<const __half*>(cosTab), static_cast<const __half*>(sinTab),
        static_cast<__half*>(out), heads, seq, headDim);
  } else {
    rotaryEmbeddingKernel<float><<<grid, block, 0, stream>>>(
        static_cast<const float*>(x), static_cast<const float*>(cosTab), static_cast<const float*>(sinTab),
        static_cast<float*>(out), heads, seq, headDim);
  }
}

}  // namespace plugins
}  // namespace tensorrt_wan
