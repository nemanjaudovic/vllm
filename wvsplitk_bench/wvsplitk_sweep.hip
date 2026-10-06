// Config sweep for vllm's wvSplitK (bf16, any N=1..5) on gfx11/gfx12, using vllm's own kernels
// (vllm_kernels.inc, produced by extract_kernels.sh from csrc/rocm/skinny_gemms.cu).
//
// Each variant = (DOT, YTILE, UNRL, WVPRGRP). Kernel choice (sml / hf / big) follows vllm's
// WVSPLITK_CFG rule: N*K fits LDS and M%YTILE==0 -> sml; N*K <= 1.2*LDS -> hf; else big.
// "vllm" row = what vllm's current gfx1x heuristic picks for that shape (16 waves, fp32 dot).
//
//   ./build_sweep.sh <N>        -> ./wvsplitk_sweep_n<N>
//   ./wvsplitk_sweep_n<N> [--shapes MxK[xcalls],...] [--rounds 3] [--csv out.csv] [--cu C]
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

#ifndef SWEEP_N
  #define SWEEP_N 1
#endif

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

struct Variant {
  std::string name;
  bool dot;
  int yt, u, wv;
  Launcher launch;
};

template <bool ND, int YT, int U>
void add_wv(std::vector<Variant>& v) {
  v.push_back({"", ND, YT, U, 2, make_launcher<ND, YT, U, 2, SWEEP_N>()});
  v.push_back({"", ND, YT, U, 4, make_launcher<ND, YT, U, 4, SWEEP_N>()});
  v.push_back({"", ND, YT, U, 8, make_launcher<ND, YT, U, 8, SWEEP_N>()});
  v.push_back({"", ND, YT, U, 16, make_launcher<ND, YT, U, 16, SWEEP_N>()});
}
template <bool ND, int YT>
void add_u(std::vector<Variant>& v) {
  add_wv<ND, YT, 1>(v); add_wv<ND, YT, 2>(v); add_wv<ND, YT, 4>(v); add_wv<ND, YT, 8>(v);
}
template <bool ND>
void add_yt(std::vector<Variant>& v) {
  add_u<ND, 1>(v); add_u<ND, 2>(v); add_u<ND, 3>(v); add_u<ND, 4>(v);
}
std::vector<Variant> make_variants() {
  std::vector<Variant> v;
  add_yt<false>(v);
  add_yt<true>(v);
  for (auto& x : v)
    x.name = std::string(x.dot ? "nd" : "fp") + "_y" + std::to_string(x.yt) + "u" +
             std::to_string(x.u) + "w" + std::to_string(x.wv);
  return v;
}

// vllm's current gfx1x (non-gfx1151) choice: WVSPLIT_TILE_CFG(32, 16, sYT, N), fp32 dot
std::string vllm_default(int M, int K, int N, int cu) {
  int sYT = (M + cu * 4 - 1) / (cu * 4);
  bool fit_lds = K * N <= MAX_LDS_LEN;
  int yt, u;
  if (sYT <= 1) yt = 1, u = 4;
  else if (N == 1 || !fit_lds || sYT <= 8) yt = 2, u = 2;
  else if (sYT <= 12) yt = 3, u = 2;
  else if (N == 4) yt = 4, u = 1;
  else yt = 4, u = 2;
  return "fp_y" + std::to_string(yt) + "u" + std::to_string(u) + "w16";
}

__global__ void ref_gemv(const bf16* W, const bf16* x, float* y, int M, int K) {
  __shared__ float red[256];
  int row = blockIdx.x, n = blockIdx.y;
  float acc = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    acc += __bfloat162float(W[(size_t)row * K + k]) * __bfloat162float(x[(size_t)n * K + k]);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) y[(size_t)n * M + row] = red[0];
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

int main(int argc, char** argv) {
  const int N = SWEEP_N;
  // MxK x calls-per-decode-step (Qwen3.5-9B; replace with your model's shapes)
  std::vector<std::tuple<int, int, int>> shapes = {
      {248320, 4096, 1}, {24576, 4096, 32}, {12288, 4096, 24},
      {10240, 4096, 8},  {4096, 12288, 32}, {4096, 4096, 32}};
  int rounds = 3, rot_mb = 1536, cu = 0;
  std::string csv = "sweep_n" + std::to_string(N) + ".csv", only;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nxt = [&]() { return std::string(argv[++i]); };
    if (a == "--rounds") rounds = atoi(nxt().c_str());
    else if (a == "--rot-mb") rot_mb = atoi(nxt().c_str());
    else if (a == "--cu") cu = atoi(nxt().c_str());
    else if (a == "--csv") csv = nxt();
    else if (a == "--only") only = nxt();  // comma list of variant names
    else if (a == "--shapes") {
      shapes.clear();
      for (auto& s : split(nxt(), ',')) {
        auto p = split(s, 'x');
        shapes.emplace_back(atoi(p[0].c_str()), atoi(p[1].c_str()),
                            p.size() > 2 ? atoi(p[2].c_str()) : 1);
      }
    } else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
  }
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  if (!cu) cu = prop.multiProcessorCount;  // = vllm num_compute_units()
  auto vars = make_variants();
  if (!only.empty()) {
    auto keep = split(only, ',');
    vars.erase(std::remove_if(vars.begin(), vars.end(), [&](const Variant& v) {
                 return std::find(keep.begin(), keep.end(), v.name) == keep.end();
               }), vars.end());
  }
  printf("device %s (%s), CuCount=%d, N=%d, %zu variants, %d rounds -> %s\n", prop.name,
         prop.gcnArchName, cu, N, vars.size(), rounds, csv.c_str());
  FILE* f = fopen(csv.c_str(), "w");
  fprintf(f, "arch,N,M,K,calls,variant,dot,ytile,unrl,waves,kernel,is_vllm_default,us,gbps,max_rel_err\n");
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));

  for (auto [M, K, calls] : shapes) {
    size_t n = (size_t)M * K, bytes = n * 2;
    int nb = std::max<int>(2, (int)((size_t)rot_mb * (1 << 20) / bytes) + 1);
    std::vector<bf16*> W(nb);
    for (int b = 0; b < nb; b++) {
      CHECK(hipMalloc(&W[b], bytes));
      fill_rand<<<1024, 256>>>(W[b], n, 1234 + b);
    }
    bf16 *x, *y; float* yref;
    CHECK(hipMalloc(&x, (size_t)N * K * 2)); CHECK(hipMalloc(&y, (size_t)N * M * 2));
    CHECK(hipMalloc(&yref, (size_t)N * M * 4));
    fill_rand<<<64, 256>>>(x, (size_t)N * K, 99);
    ref_gemv<<<dim3(M, N), 256>>>(W[0], x, yref, M, K);
    std::vector<float> href((size_t)N * M);
    std::vector<bf16> hy((size_t)N * M);
    CHECK(hipMemcpy(href.data(), yref, href.size() * 4, hipMemcpyDeviceToHost));
    double scale = 0;
    for (float v : href) scale = std::max(scale, (double)fabs(v));
    std::string def = vllm_default(M, K, N, cu);

    // correctness + rough time per call (to size the timed loop)
    std::vector<double> err(vars.size(), -1), est(vars.size());
    for (size_t i = 0; i < vars.size(); i++) {
      CHECK(hipMemset(y, 0xff, hy.size() * 2));
      vars[i].launch(W[0], x, y, M, K, cu);
      if (hipGetLastError() != hipSuccess) continue;
      CHECK(hipMemcpy(hy.data(), y, hy.size() * 2, hipMemcpyDeviceToHost));
      double e = 0;
      for (size_t j = 0; j < hy.size(); j++)
        e = std::max(e, (double)fabs(__bfloat162float(hy[j]) - href[j]));
      err[i] = std::isnan(e) ? 1e9 : e / scale;
      CHECK(hipEventRecord(e0));
      for (int it = 0; it < 3; it++) vars[i].launch(W[(it + 1) % nb], x, y, M, K, cu);
      CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
      float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
      est[i] = ms / 3;
    }
    std::vector<std::vector<double>> t(vars.size());
    for (int r = 0; r < rounds; r++)  // rounds interleave variants to cancel thermal/clock drift
      for (size_t i = 0; i < vars.size(); i++) {
        if (err[i] < 0) continue;
        int iters = std::clamp((int)(20.0 / std::max(est[i], 1e-3)), 10, 200);  // ~20 ms
        int c = 0;
        for (int w = 0; w < 5; w++) vars[i].launch(W[c++ % nb], x, y, M, K, cu);
        CHECK(hipEventRecord(e0));
        for (int it = 0; it < iters; it++) vars[i].launch(W[c++ % nb], x, y, M, K, cu);
        CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
        float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
        t[i].push_back(ms * 1e3 / iters);
      }
    double def_us = 0, best_us = 1e30;
    std::string best;
    for (size_t i = 0; i < vars.size(); i++) {
      if (t[i].empty()) continue;
      std::sort(t[i].begin(), t[i].end());
      double us = t[i][t[i].size() / 2];
      auto& v = vars[i];
      bool ok = err[i] < 1e-2;
      fprintf(f, "%s,%d,%d,%d,%d,%s,%d,%d,%d,%d,%s,%d,%.2f,%.1f,%.2e\n", prop.gcnArchName, N, M, K,
              calls, v.name.c_str(), v.dot, v.yt, v.u, v.wv, kernel_for(M, K, N, v.yt),
              v.name == def, us, bytes / us / 1e3, err[i]);
      if (v.name == def) def_us = us;
      if (ok && us < best_us) best_us = us, best = v.name;
      if (!ok) printf("  WRONG RESULT %s %dx%d err %.2e\n", v.name.c_str(), M, K, err[i]);
    }
    printf("%7dx%-6d N=%d  vllm(%s) %8.1f us  best %-14s %8.1f us  %+5.1f%%  [kernel %s]\n", M, K,
           N, def.c_str(), def_us, best.c_str(), best_us, 100 * (best_us - def_us) / def_us,
           kernel_for(M, K, N, 2));
    fflush(f);
    for (auto p : W) CHECK(hipFree(p));
    CHECK(hipFree(x)); CHECK(hipFree(y)); CHECK(hipFree(yref));
  }
  fclose(f);
  return 0;
}
