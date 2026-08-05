#include "adalayernorm/plugin.h"
#include <cuda_fp16.h>

namespace tensorrt_wan {
namespace plugins {

// One block per (b, s) row; a block-wide reduction computes mean/variance over the C channels,
// then every thread normalizes and applies the per-batch (scale, shift) modulation.
template <typename T>
__global__ void adaLayerNormKernel(const T* __restrict__ x, const T* __restrict__ scale,
                                    const T* __restrict__ shift, T* __restrict__ out, int seq, int channels,
                                    float eps) {
  extern __shared__ float shared[];
  float* sumBuf = shared;
  float* sqSumBuf = shared + blockDim.x;

  const int b = blockIdx.x;
  const int s = blockIdx.y;
  const long rowBase = ((long)b * seq + s) * channels;
  const long modBase = (long)b * channels;

  float sum = 0.f, sqSum = 0.f;
  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    const float v = static_cast<float>(x[rowBase + c]);
    sum += v;
    sqSum += v * v;
  }
  sumBuf[threadIdx.x] = sum;
  sqSumBuf[threadIdx.x] = sqSum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sumBuf[threadIdx.x] += sumBuf[threadIdx.x + stride];
      sqSumBuf[threadIdx.x] += sqSumBuf[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float mean = sumBuf[0] / channels;
  const float var = sqSumBuf[0] / channels - mean * mean;
  const float invStd = rsqrtf(var + eps);

  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    const float normed = (static_cast<float>(x[rowBase + c]) - mean) * invStd;
    const float sc = static_cast<float>(scale[modBase + c]);
    const float sh = static_cast<float>(shift[modBase + c]);
    out[rowBase + c] = static_cast<T>(normed * (1.f + sc) + sh);
  }
}

void launchAdaLayerNorm(const void* x, const void* scale, const void* shift, void* out, int batch, int seq,
                         int channels, float eps, bool isHalf, cudaStream_t stream) {
  const int threads = std::min(channels, 256);
  const dim3 grid(batch, seq);
  const size_t sharedBytes = 2 * threads * sizeof(float);

  if (isHalf) {
    adaLayerNormKernel<__half><<<grid, threads, sharedBytes, stream>>>(
        static_cast<const __half*>(x), static_cast<const __half*>(scale), static_cast<const __half*>(shift),
        static_cast<__half*>(out), seq, channels, eps);
  } else {
    adaLayerNormKernel<float><<<grid, threads, sharedBytes, stream>>>(
        static_cast<const float*>(x), static_cast<const float*>(scale), static_cast<const float*>(shift),
        static_cast<float*>(out), seq, channels, eps);
  }
}

}  // namespace plugins
}  // namespace tensorrt_wan
