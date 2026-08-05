#include "patch_reconstruct/plugin.h"
#include <cuda_fp16.h>

namespace tensorrt_wan {
namespace plugins {

// One thread per output voxel (b, c, t, h, w): looks up which token covers it and which
// patch-local output channel it corresponds to, then dot-products that token's embedding with
// the matching weight column.
template <typename T>
__global__ void patchReconstructKernel(const T* __restrict__ tokens, const T* __restrict__ weight,
                                        const T* __restrict__ bias, T* __restrict__ out, int numTokens, int embedDim,
                                        int channels, int patchT, int patchH, int patchW, int nH, int nW) {
  const int outH = nH * patchH, outW = nW * patchW;
  const long totalVoxels = (long)channels * (numTokens / (nH * nW)) * patchT * outH * outW;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalVoxels) return;

  const int w = idx % outW;
  const int h = (idx / outW) % outH;
  const long tmp = idx / ((long)outW * outH);
  const int t = tmp % (numTokens / (nH * nW) * patchT);
  const long tmp2 = tmp / (numTokens / (nH * nW) * patchT);
  const int c = tmp2 % channels;
  const int b = tmp2 / channels;

  const int tIdx = t / patchT, pt = t % patchT;
  const int hIdx = h / patchH, ph = h % patchH;
  const int wIdx = w / patchW, pw = w % patchW;
  const int tokenIdx = (tIdx * nH + hIdx) * nW + wIdx;
  const int patchSize = channels * patchT * patchH * patchW;
  const int k = ((c * patchT + pt) * patchH + ph) * patchW + pw;

  const long tokenBase = ((long)b * numTokens + tokenIdx) * embedDim;
  float acc = static_cast<float>(bias[k]);
  for (int e = 0; e < embedDim; ++e) {
    acc += static_cast<float>(tokens[tokenBase + e]) * static_cast<float>(weight[(long)k * embedDim + e]);
  }
  out[idx] = static_cast<T>(acc);
}

void launchPatchReconstruct(const void* tokens, const void* weight, const void* bias, void* out, int batch,
                             int numTokens, int embedDim, int channels, int patchT, int patchH, int patchW, int nH,
                             int nW, bool isHalf, cudaStream_t stream) {
  const int nT = numTokens / (nH * nW);
  const long totalVoxels = (long)batch * channels * nT * patchT * nH * patchH * nW * patchW;
  const int threads = 256;
  const long blocks = (totalVoxels + threads - 1) / threads;

  if (isHalf) {
    patchReconstructKernel<__half><<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(tokens), static_cast<const __half*>(weight), static_cast<const __half*>(bias),
        static_cast<__half*>(out), numTokens, embedDim, channels, patchT, patchH, patchW, nH, nW);
  } else {
    patchReconstructKernel<float><<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(tokens), static_cast<const float*>(weight), static_cast<const float*>(bias),
        static_cast<float*>(out), numTokens, embedDim, channels, patchT, patchH, patchW, nH, nW);
  }
}

}  // namespace plugins
}  // namespace tensorrt_wan
