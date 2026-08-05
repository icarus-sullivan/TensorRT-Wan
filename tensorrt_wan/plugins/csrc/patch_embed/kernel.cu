#include "patch_embed/plugin.h"
#include <cuda_fp16.h>

namespace tensorrt_wan {
namespace plugins {

// One block per (batch, token, embedDim-chunk); each thread accumulates one output channel's
// dot product over the flattened patch (C * patchT * patchH * patchW elements).
template <typename T>
__global__ void patchEmbedKernel(const T* __restrict__ latents, const T* __restrict__ weight,
                                  const T* __restrict__ bias, T* __restrict__ out, int channels, int t, int h, int w,
                                  int patchT, int patchH, int patchW, int embedDim) {
  const int nH = h / patchH, nW = w / patchW;
  const int patchSize = channels * patchT * patchH * patchW;

  const int b = blockIdx.z;
  const int tokenIdx = blockIdx.y;
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= embedDim) return;

  const int wIdx = tokenIdx % nW;
  const int hIdx = (tokenIdx / nW) % nH;
  const int tIdx = tokenIdx / (nW * nH);

  float acc = static_cast<float>(bias[e]);
  int k = 0;
  for (int c = 0; c < channels; ++c) {
    for (int pt = 0; pt < patchT; ++pt) {
      const int srcT = tIdx * patchT + pt;
      for (int ph = 0; ph < patchH; ++ph) {
        const int srcH = hIdx * patchH + ph;
        for (int pw = 0; pw < patchW; ++pw, ++k) {
          const int srcW = wIdx * patchW + pw;
          const long srcIdx = ((((long)b * channels + c) * t + srcT) * h + srcH) * w + srcW;
          acc += static_cast<float>(latents[srcIdx]) * static_cast<float>(weight[(long)e * patchSize + k]);
        }
      }
    }
  }

  const long numTokens = (long)(t / patchT) * nH * nW;
  out[((long)b * numTokens + tokenIdx) * embedDim + e] = static_cast<T>(acc);
}

void launchPatchEmbed(const void* latents, const void* weight, const void* bias, void* out, int batch, int channels,
                       int t, int h, int w, int patchT, int patchH, int patchW, int embedDim, bool isHalf,
                       cudaStream_t stream) {
  const int numTokens = (t / patchT) * (h / patchH) * (w / patchW);
  const dim3 block(std::min(embedDim, 256));
  const dim3 grid((embedDim + block.x - 1) / block.x, numTokens, batch);

  if (isHalf) {
    patchEmbedKernel<__half><<<grid, block, 0, stream>>>(
        static_cast<const __half*>(latents), static_cast<const __half*>(weight), static_cast<const __half*>(bias),
        static_cast<__half*>(out), channels, t, h, w, patchT, patchH, patchW, embedDim);
  } else {
    patchEmbedKernel<float><<<grid, block, 0, stream>>>(
        static_cast<const float*>(latents), static_cast<const float*>(weight), static_cast<const float*>(bias),
        static_cast<float*>(out), channels, t, h, w, patchT, patchH, patchW, embedDim);
  }
}

}  // namespace plugins
}  // namespace tensorrt_wan
