#include "time_embedding/plugin.h"
#include <cuda_fp16.h>

namespace tensorrt_wan {
namespace plugins {

// timestep is always fp32 regardless of model precision (it's a scalar per batch element, not a
// throughput-sensitive tensor); only the output embedding is cast to the model's compute dtype.
template <typename T>
__global__ void timeEmbeddingKernel(const float* __restrict__ timestep, T* __restrict__ out, int dim,
                                     float maxPeriod) {
  const int half = dim / 2;
  const int b = blockIdx.x;
  const int i = threadIdx.x;
  if (i >= half) return;

  const float freq = expf(-logf(maxPeriod) * i / half);
  const float arg = timestep[b] * freq;
  out[(long)b * dim + i] = static_cast<T>(cosf(arg));
  out[(long)b * dim + half + i] = static_cast<T>(sinf(arg));
}

void launchTimeEmbedding(const void* timestep, void* out, int batch, int dim, float maxPeriod, bool isHalf,
                          cudaStream_t stream) {
  const int half = dim / 2;
  if (isHalf) {
    timeEmbeddingKernel<__half>
        <<<batch, half, 0, stream>>>(static_cast<const float*>(timestep), static_cast<__half*>(out), dim, maxPeriod);
  } else {
    timeEmbeddingKernel<float>
        <<<batch, half, 0, stream>>>(static_cast<const float*>(timestep), static_cast<float*>(out), dim, maxPeriod);
  }
}

}  // namespace plugins
}  // namespace tensorrt_wan
