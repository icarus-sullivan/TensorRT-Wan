#include "activation/plugin.h"
#include <cuda_fp16.h>

namespace tensorrt_wan {
namespace plugins {

__device__ __forceinline__ float siluf(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float geluf(float x) {
  return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float quickGeluf(float x) { return x / (1.f + expf(-1.702f * x)); }

template <typename T>
__global__ void activationKernel(ActivationKind kind, const T* __restrict__ x, T* __restrict__ out, long numel) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  const float v = static_cast<float>(x[idx]);
  float r;
  switch (kind) {
    case ActivationKind::kGELU: r = geluf(v); break;
    case ActivationKind::kQuickGELU: r = quickGeluf(v); break;
    default: r = siluf(v); break;
  }
  out[idx] = static_cast<T>(r);
}

void launchActivation(ActivationKind kind, const void* x, void* out, long numel, bool isHalf, cudaStream_t stream) {
  const int threads = 256;
  const long blocks = (numel + threads - 1) / threads;
  if (isHalf) {
    activationKernel<__half>
        <<<blocks, threads, 0, stream>>>(kind, static_cast<const __half*>(x), static_cast<__half*>(out), numel);
  } else {
    activationKernel<float>
        <<<blocks, threads, 0, stream>>>(kind, static_cast<const float*>(x), static_cast<float*>(out), numel);
  }
}

}  // namespace plugins
}  // namespace tensorrt_wan
