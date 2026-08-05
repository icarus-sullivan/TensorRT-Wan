#include "video_ops/plugin.h"
#include <cuda_fp16.h>

namespace tensorrt_wan {
namespace plugins {

template <typename T>
__global__ void temporalResizeKernel(const T* __restrict__ x, T* __restrict__ out, int channels, int t, int h, int w,
                                      int factor) {
  const long outT = (long)t * factor;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long hw = (long)h * w;
  const long total = (long)channels * outT * hw;
  if (idx >= total) return;

  const long spatial = idx % hw;
  const long tOut = (idx / hw) % outT;
  const long c = idx / (hw * outT);
  const long tIn = tOut / factor;  // nearest-neighbor

  out[idx] = x[(c * t + tIn) * hw + spatial];
}

void launchTemporalResize(const void* x, void* out, int batch, int channels, int t, int h, int w, int factor,
                           bool isHalf, cudaStream_t stream) {
  const long perBatch = (long)channels * t * factor * h * w;
  const int threads = 256;
  const long blocks = (perBatch + threads - 1) / threads;

  for (int b = 0; b < batch; ++b) {
    if (isHalf) {
      const auto* in = static_cast<const __half*>(x) + (long)b * channels * t * h * w;
      auto* o = static_cast<__half*>(out) + (long)b * perBatch;
      temporalResizeKernel<__half><<<blocks, threads, 0, stream>>>(in, o, channels, t, h, w, factor);
    } else {
      const auto* in = static_cast<const float*>(x) + (long)b * channels * t * h * w;
      auto* o = static_cast<float*>(out) + (long)b * perBatch;
      temporalResizeKernel<float><<<blocks, threads, 0, stream>>>(in, o, channels, t, h, w, factor);
    }
  }
}

}  // namespace plugins
}  // namespace tensorrt_wan
