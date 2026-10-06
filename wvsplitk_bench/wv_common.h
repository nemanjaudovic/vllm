// Shared by wvsplitk_sweep.hip and wvsplitk_accuracy.hip: vllm kernels (fp32-dot and native-dot
// builds), vllm host-side dispatch rules, launch helpers.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include <string>

#define CHECK(x)                                                                    \
  do {                                                                              \
    hipError_t e_ = (x);                                                            \
    if (e_ != hipSuccess) {                                                         \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_));     \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

// ---- prelude: what vllm_kernels.inc expects from the top of skinny_gemms.cu ----
#if defined(__GFX11__) || defined(__GFX12__)
  #define __HIP__GFX1X__
#endif
#define LDS_SIZE 64 * 1024  // gfx11/gfx12
#define UNREACHABLE_CODE assert(false);
template <typename T>
__device__ __forceinline__ T __float2s(float v);
template <>
__device__ __forceinline__ half __float2s(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ __hip_bfloat16 __float2s(float v) { return __float2bfloat16(v); }
template <typename T>
__device__ __forceinline__ T loadnt(T* addr) { return __builtin_nontemporal_load(addr); }
typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));

// ---- vllm kernels, twice: current fp32 bf16 dot, and native v_dot2_f32_bf16 ----
namespace fp32dot {
#define DOT2C(V0, V2, V3)                                               \
  if constexpr (std::is_same_v<scalar_t, half>) {                       \
    asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(V0) : "v"(V2), "v"(V3)); \
  } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {      \
    float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *       \
               __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));        \
    V0 += (s.x + s.y);                                                  \
  }
#include "vllm_kernels.inc"
#undef DOT2C
}  // namespace fp32dot

namespace nativedot {
#define DOT2C(V0, V2, V3)                                                               \
  if constexpr (std::is_same_v<scalar_t, half>) {                                       \
    asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(V0) : "v"(V2), "v"(V3));                 \
  } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {                      \
    V0 = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2_t, V2),              \
                                         __builtin_bit_cast(bf16x2_t, V3), V0, false); \
  }
#include "vllm_kernels.inc"
#undef DOT2C
}  // namespace nativedot

using bf16 = __hip_bfloat16;
constexpr int MAX_LDS_LEN = LDS_SIZE / 2;  // elements, as in vllm host code

// vllm's mindiv with a guard for WvPrGrp < 13 (vllm's version divides by <= 0 there)
int mindiv_safe(int N, int div1, int div2) {
  int nPrRnd = div1 * div2, n = std::min(13, div2), rnds[13];
  for (int i = 0; i < n; i++) {
    rnds[i] = (N + nPrRnd - 1) / nPrRnd;
    nPrRnd -= div1;
  }
  for (int i = n - 1; i >= 0; i--)
    if (rnds[0] == rnds[i]) return div2 - i;
  return 0;
}

const char* kernel_for(int M, int K, int N, int YT) {
  if (K * N <= MAX_LDS_LEN && M % YT == 0) return "sml";
  if (K * N <= MAX_LDS_LEN * 1.2) return "hf";
  return "big";
}

// W: MxK weight, x: NxK activation, y: NxM output
using Launcher = std::function<void(const bf16* W, const bf16* x, bf16* y, int M, int K, int cu)>;

template <bool ND, int YT, int U, int WV, int N>
Launcher make_launcher() {
  return [](const bf16* W, const bf16* x, bf16* y, int M, int K, int cu) {
    int wv = mindiv_safe(M, cu * YT, WV);
    dim3 grid(cu), block(32, WV);
    std::string k = kernel_for(M, K, N, YT);
    // args as in vllm: (K, Kap=W stride, Kbp=x stride, M, Bx, By, B=W, A=x, BIAS, C, wv, cu)
#define LAUNCH(NS)                                                                              \
  if (k == "sml")                                                                               \
    NS::wvSplitK_hf_sml_<bf16, 32, YT, WV, 8, U, N>                                             \
        <<<grid, block>>>(K, K, K, M, 1, 1, W, x, nullptr, y, wv, cu);                          \
  else if (k == "hf")                                                                           \
    NS::wvSplitK_hf_<bf16, 32, YT, WV, 8, U, N>                                                 \
        <<<grid, block>>>(K, K, K, M, 1, 1, W, x, nullptr, y, wv, cu);                          \
  else                                                                                          \
    NS::wvSplitK_hf_big_<bf16, 32, YT, WV, 8, U, N>                                             \
        <<<grid, block>>>(K, K, K, M, 1, 1, W, x, nullptr, y, wv, cu);
    if constexpr (ND) { LAUNCH(nativedot) } else { LAUNCH(fp32dot) }
#undef LAUNCH
  };
}

// vllm's current choice (WVSPLIT_TILE in skinny_gemms.cu), always 16 waves + fp32 dot:
// gfx1151 has its own rule list (PR #40784); other gfx1x use WVSPLIT_TILE_CFG(32, 16, sYT, N).
std::string vllm_default(int M, int K, int N, int cu, bool gfx1151) {
  int sYT = (M + cu * 4 - 1) / (cu * 4);
  bool fit_lds = K * N <= MAX_LDS_LEN;
  int yt, u;
  if (gfx1151) {
    if (sYT <= 1) yt = 1, u = 4;
    else if (K % 1024 == 512 && K >= 1536 && (sYT >= 40 || K >= 4096)) yt = 4, u = 1;
    else if (K < 1024) yt = 2, u = 4;
    else if (K <= 2048 && (N >= 2 || sYT <= 26)) yt = 1, u = 4;
    else if (N >= 2 && !fit_lds) yt = 1, u = 4;
    else if (N == 1) yt = 1, u = 2;
    else yt = 1, u = 1;
  } else if (sYT <= 1) yt = 1, u = 4;
  else if (N == 1 || !fit_lds || sYT <= 8) yt = 2, u = 2;
  else if (sYT <= 12) yt = 3, u = 2;
  else if (N == 4) yt = 4, u = 1;
  else yt = 4, u = 2;
  return "fp_y" + std::to_string(yt) + "u" + std::to_string(u) + "w16";
}

__global__ void fill_rand(bf16* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    p[i] = __float2bfloat16(((h & 0xffff) / 65535.f - 0.5f) * 0.1f);
  }
}

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  size_t a = 0, b;
  while ((b = s.find(c, a)) != std::string::npos) { out.push_back(s.substr(a, b - a)); a = b + 1; }
  if (a < s.size()) out.push_back(s.substr(a));
  return out;
}
