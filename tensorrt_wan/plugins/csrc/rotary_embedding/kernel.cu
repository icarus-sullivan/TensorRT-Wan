#include "rotary_embedding/plugin.h"
#include <cuda_fp16.h>

namespace tensorrt_wan {
namespace plugins {

// One thread per (b, h, s, p) with pair index p in [0, headDim/2) — each thread rotates one
// *adjacent* pair (x[2p], x[2p+1]), matching Wan's interleaved-pair convention described in
// plugin.h (not the rotate-half split this kernel implemented previously).
template <typename T>
__global__ void rotaryEmbeddingKernel(const T* __restrict__ x, const T* __restrict__ cosTab,
                                       const T* __restrict__ sinTab, T* __restrict__ out, int heads, int seq,
                                       int headDim) {
  const int half = headDim / 2;
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= half) return;

  const int s = blockIdx.y;
  const int bh = blockIdx.z;  // flattened (batch, head) index

  const long base = (long)bh * seq * headDim + (long)s * headDim;
  const long cosSinBase = (long)s * headDim;

  const float x1 = static_cast<float>(x[base + 2 * p]);
  const float x2 = static_cast<float>(x[base + 2 * p + 1]);
  // cos/sin tables are pre-duplicated per pair (cos[2p] == cos[2p+1]); either index works.
  const float c = static_cast<float>(cosTab[cosSinBase + 2 * p]);
  const float sn = static_cast<float>(sinTab[cosSinBase + 2 * p]);

  out[base + 2 * p] = static_cast<T>(x1 * c - x2 * sn);
  out[base + 2 * p + 1] = static_cast<T>(x1 * sn + x2 * c);
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
