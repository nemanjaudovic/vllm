// Standalone wvSplitK (N=1, bf16) harness for gfx11: correctness + timing, no vllm/torch.
//
//   ./build.sh && ./wvsplitk_bench [--variants base,d1p0y2u2,...] [--shapes MxK,...] [--iters N]
//
// Variants:
//   base         verbatim copy of vllm wvSplitK_hf_sml_<bf16,32,YT,16,8,UNRL,1> (default YT=2,UNRL=2)
//   d<D>p<P>y<Y>u<U>   wv_opt: D=1 native v_dot2_f32_bf16, P=1 software-pipelined weight loads
//                      (next tile's loads issued before current tile's math, also across row tiles)
// Weights are rotated over >= --rot-mb MB of buffers so every call streams from DRAM.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <functional>
#include <random>
#include <string>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                        \
    hipError_t e_ = (x);                                                      \
    if (e_ != hipSuccess) {                                                   \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define LDS_SIZE 64 * 1024
using bf16 = __hip_bfloat16;

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}
__device__ inline unsigned int min__(uint32_t a, uint32_t b) { return min(a, b); }

// ---------------------------------------------------------------------------
// BASE: copy of vllm csrc/rocm/skinny_gemms.cu wvSplitK_hf_sml_ (gfx1x, bf16 path only)
// ---------------------------------------------------------------------------
#define DOT2C(V0, V2, V3)                                         \
  {                                                               \
    float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) * \
               __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));  \
    V0 += (s.x + s.y);                                            \
  }

typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));
// proposed vllm gfx1x bf16 DOT2C
#define DOT2C_NATIVE(V0, V2, V3)                                                         \
  V0 = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2_t, V2),                 \
                                       __builtin_bit_cast(bf16x2_t, V3), V0, false);

template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N,
          bool NATIVE_DOT = false>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_sml_(const int K, const int Kbp, const int Kap, const int M, const int Bx,
                     const int By, const scalar_t* B, const scalar_t* __restrict__ A,
                     const scalar_t* __restrict__ BIAS, scalar_t* C, const int _WvPrGrp,
                     const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  using scalar8 = __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    scalar8 h8;
  };
  __shared__ scalar_t s[max_lds_len];
  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK; k < min__(Kap * N, max_lds_len);
       k += THRDS * WvPrGrp * A_CHUNK) {
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  }
  __syncthreads();
  if (threadIdx.y >= _WvPrGrp) return;
  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;
  while (m < M) {
    float sum[N][YTILE] = {};
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
      }
      for (uint32_t k2 = 0; k2 < UNRL; k2++)
        for (uint32_t n = 0; n < N; n++)
          for (int y = 0; y < YTILE; y++)
            for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
              if constexpr (NATIVE_DOT) {
                DOT2C_NATIVE(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              } else {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            }
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int n = 0; n < N; n++)
      for (int y = 0; y < YTILE; y++) {
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf, 1);
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf, 1);
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf, 1);
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf, 1);
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    if (threadIdx.x == (THRDS - 1)) {
      for (int n = 0; n < N; n++)
        for (int y = 0; y < YTILE; y++) {
          if (BIAS) sum[n][y] += __bfloat162float(BIAS[(m + y) % Bx + (n % By) * Bx]);
          C[m + y + n * M] = __float2bfloat16(sum[n][y]);
        }
    }
    m += CuCount * _WvPrGrp * YTILE;
  }
}

// ---------------------------------------------------------------------------
// OPT: N=1, bf16, no bias. BF16DOT: native dot. PIPE: register double-buffered weights.
// ---------------------------------------------------------------------------
template <bool BF16DOT>
__device__ __forceinline__ float dot2(float acc, float a, float b) {
  if constexpr (BF16DOT) {
    return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2_t, a),
                                           __builtin_bit_cast(bf16x2_t, b), acc, false);
  } else {
    DOT2C(acc, a, b)
    return acc;
  }
}

template <int THRDS, int YTILE, int WvPrGrp, int UNRL, bool BF16DOT, bool PIPE,
          int LDS_ELEMS = LDS_SIZE / 2>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wv_opt(const int K, const int Kbp, const int M, const bf16* B, const bf16* __restrict__ A,
           bf16* C, const int _WvPrGrp, const int CuCount) {
  constexpr int A_CHUNK = 8;
  constexpr int max_lds_len = LDS_ELEMS;  // smaller -> more WGs/WGP can be resident
  constexpr uint32_t KSTEP = THRDS * A_CHUNK * UNRL;
  using f4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  union bigType {
    float f[4];
    f4 h8;
  };
  __shared__ bf16 s[max_lds_len];
  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK; k < min__(K, max_lds_len);
       k += THRDS * WvPrGrp * A_CHUNK)
    *((bigType*)(&s[k])) = *((const bigType*)(&A[k]));
  __syncthreads();
  // readfirstlane: threadIdx.y is wave-uniform but lives in a VGPR; without this the compiler
  // treats the m/k loop as divergent and the waitcnt pass drains vmcnt(0) at the loop header.
  const uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.y);
  if (wave >= (uint32_t)_WvPrGrp) return;

  const uint32_t mstride = CuCount * _WvPrGrp * YTILE;
  uint32_t m = (blockIdx.x * _WvPrGrp + wave) * YTILE;

  auto load_w = [&](bigType (&w)[YTILE][UNRL], uint32_t mm, uint32_t k1) {
#pragma unroll
    for (int k2 = 0; k2 < UNRL; k2++) {
      uint32_t k_ = k1 + k2 * THRDS * A_CHUNK + threadIdx.x * A_CHUNK;
      const bf16* B_ = &B[min__(k_, K - A_CHUNK)];
#pragma unroll
      for (int y = 0; y < YTILE; y++)
        w[y][k2].h8 = loadnt((f4*)(&B_[(size_t)min__(y + mm, M - 1) * Kbp]));
    }
  };
  auto compute = [&](float (&sum)[YTILE], bigType (&w)[YTILE][UNRL], uint32_t k1) {
#pragma unroll
    for (int k2 = 0; k2 < UNRL; k2++) {
      uint32_t k_ = k1 + k2 * THRDS * A_CHUNK + threadIdx.x * A_CHUNK;
      bigType a;
      if (k_ < K) a = *((const bigType*)(&s[k_]));
      else a.h8 = f4{0, 0, 0, 0};
#pragma unroll
      for (int y = 0; y < YTILE; y++)
#pragma unroll
        for (int b = 0; b < 4; b++) sum[y] = dot2<BF16DOT>(sum[y], a.f[b], w[y][k2].f[b]);
    }
  };
  auto finish = [&](float (&sum)[YTILE], uint32_t mm) {
#pragma unroll
    for (int y = 0; y < YTILE; y++) {
      sum[y] += __builtin_amdgcn_mov_dpp(sum[y], 0x118, 0xf, 0xf, 1);
      sum[y] += __builtin_amdgcn_mov_dpp(sum[y], 0x114, 0xf, 0xf, 1);
      sum[y] += __builtin_amdgcn_mov_dpp(sum[y], 0x112, 0xf, 0xf, 1);
      sum[y] += __builtin_amdgcn_mov_dpp(sum[y], 0x111, 0xf, 0xf, 1);
      sum[y] += __shfl_xor(sum[y], 16);
    }
    if (threadIdx.x == (THRDS - 1))
#pragma unroll
      for (int y = 0; y < YTILE; y++) C[mm + y] = __float2bfloat16(sum[y]);
  };

  if constexpr (!PIPE) {
    while (m < M) {
      float sum[YTILE] = {};
      for (uint32_t k1 = 0; k1 < K; k1 += KSTEP) {
        bigType w[YTILE][UNRL];
        load_w(w, m, k1);
        compute(sum, w, k1);
      }
      finish(sum, m);
      m += mstride;
    }
  } else {
    // Flattened (m, k1) iteration; tile i+1's loads are in flight while tile i is computed.
    // Manually ping-ponged (w0/w1) so no register copies are needed. The trip count is computed
    // up front so the loop has a single exit: with mid-body breaks the structurizer merges paths
    // and the (path-insensitive) waitcnt pass drains the prefetch at the loop header.
    if (m >= (uint32_t)M) return;
    const uint32_t kiters = (K + KSTEP - 1) / KSTEP;
    const uint32_t T = ((M - 1 - m) / mstride + 1) * kiters;
    float sum[YTILE] = {};
    uint32_t k1 = 0;
    bigType w0[YTILE][UNRL], w1[YTILE][UNRL];
    load_w(w0, m, 0);
    // prefetch is unconditional (rows clamp to M-1): a branch around it also costs the waitcnts
    auto step = [&](bigType (&cur)[YTILE][UNRL], bigType (&nxt)[YTILE][UNRL]) {
      uint32_t mn = m, kn = k1 + KSTEP;
      if (kn >= (uint32_t)K) { kn = 0; mn += mstride; }
      load_w(nxt, mn, kn);
      compute(sum, cur, k1);
      if (kn == 0) {
        finish(sum, m);
#pragma unroll
        for (int y = 0; y < YTILE; y++) sum[y] = 0.f;
      }
      m = mn; k1 = kn;
    };
    uint32_t it = 0;
    for (; it + 2 <= T; it += 2) {
      step(w0, w1);
      step(w1, w0);
    }
    if (it < T) step(w0, w1);
  }
}

// fp32 reference: one block per row
__global__ void ref_gemv(const bf16* W, const bf16* x, float* y, int M, int K) {
  __shared__ float red[256];
  int row = blockIdx.x;
  float acc = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    acc += __bfloat162float(W[(size_t)row * K + k]) * __bfloat162float(x[k]);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) y[row] = red[0];
}

__global__ void fill_rand(bf16* p, size_t n, uint32_t seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    p[i] = __float2bfloat16(((h & 0xffff) / 65535.f - 0.5f) * 0.1f);
  }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
int mindiv(int N, int div1, int div2) {  // copy of vllm host helper
  int nPrRnd = div1 * div2;
  int rnds[13];
  const int n = std::min(13, div2);  // guard added for div2 < 13 (vllm only uses 16)
  for (int i = 0; i < n; i++) {
    rnds[i] = (N + nPrRnd - 1) / nPrRnd;
    nPrRnd -= div1;
  }
  for (int i = n - 1; i >= 0; i--)
    if (rnds[0] == rnds[i]) return (div2 - i);
  return 0;
}

using Launcher = std::function<void(const bf16* W, const bf16* x, bf16* y, int M, int K, int cu)>;

template <int YT, int U, int WV = 16, bool ND = false>
Launcher base_launcher() {
  return [](const bf16* W, const bf16* x, bf16* y, int M, int K, int cu) {
    int wv = mindiv(M, cu * YT, WV);
    wvSplitK_hf_sml_<bf16, 32, YT, WV, 8, U, 1, ND>
        <<<dim3(cu), dim3(32, WV)>>>(K, K, K, M, 1, 1, W, x, nullptr, y, wv, cu);
  };
}
template <int YT, int U, bool D, bool P>
Launcher opt_launcher() {
  return [](const bf16* W, const bf16* x, bf16* y, int M, int K, int cu) {
    int wv = mindiv(M, cu * YT, 16);
    wv_opt<32, YT, 16, U, D, P><<<dim3(cu), dim3(32, 16)>>>(K, K, M, W, x, y, wv, cu);
  };
}

// WV waves per WG, G WGs per WGP (grid = cu*G); LDS sized for K <= 12288 so 2+ WGs fit.
template <int YT, int U, bool D, bool P, int WV, int G>
Launcher occ_launcher() {
  return [](const bf16* W, const bf16* x, bf16* y, int M, int K, int cu) {
    int wv = mindiv(M, cu * G * YT, WV);
    wv_opt<32, YT, WV, U, D, P, 12288>
        <<<dim3(cu * G), dim3(32, WV)>>>(K, K, M, W, x, y, wv, cu * G);
  };
}

std::map<std::string, std::pair<int, Launcher>> make_variants() {  // name -> (YTILE, launcher)
  std::map<std::string, std::pair<int, Launcher>> v;
  v["base"] = {2, base_launcher<2, 2>()};
  v["base_y1u4"] = {1, base_launcher<1, 4>()};
  v["base_y2u4"] = {2, base_launcher<2, 4>()};
  // the exact proposed vllm change: verbatim kernel + native-dot DOT2C + (YTILE, UNRL, waves)
  v["vllm_dot_y2u2w16"] = {2, base_launcher<2, 2, 16, true>()};
  v["vllm_y2u4w4"] = {2, base_launcher<2, 4, 4, false>()};
  v["vllm_dot_y2u4w4"] = {2, base_launcher<2, 4, 4, true>()};
  v["vllm_dot_y2u4w8"] = {2, base_launcher<2, 4, 8, true>()};
#define OPT(D, P, Y, U) \
  v["d" #D "p" #P "y" #Y "u" #U] = {Y, opt_launcher<Y, U, (bool)D, (bool)P>()};
#define OPT4(Y, U) OPT(0, 0, Y, U) OPT(1, 0, Y, U) OPT(0, 1, Y, U) OPT(1, 1, Y, U)
  OPT4(1, 2) OPT4(1, 4) OPT4(2, 1) OPT4(2, 2) OPT4(2, 4) OPT4(4, 1) OPT4(4, 2)
#define OCC(Y, U, WV, G) \
  v["d1p0y" #Y "u" #U "w" #WV "g" #G] = {Y, occ_launcher<Y, U, true, false, WV, G>()};
#define OCCW(Y, U, G) OCC(Y, U, 8, G) OCC(Y, U, 16, G) OCC(Y, U, 24, G) OCC(Y, U, 32, G)
  OCCW(2, 4, 1) OCCW(2, 4, 2) OCCW(2, 4, 3) OCCW(2, 4, 4)
#define OCCK(Y, U) OCC(Y, U, 2, 1) OCC(Y, U, 4, 1) OCC(Y, U, 8, 1) OCC(Y, U, 16, 1)
  OCCK(1, 8) OCCK(2, 8) OCCK(1, 16) OCC(2, 4, 4, 1) OCC(2, 4, 2, 1) OCC(1, 4, 4, 1) OCC(4, 4, 4, 1) OCC(4, 4, 8, 1)
  OCC(4, 2, 4, 1) OCC(4, 2, 8, 1)
#define OCCP(Y, U, WV) \
  v["d1p1y" #Y "u" #U "w" #WV "g1"] = {Y, occ_launcher<Y, U, true, true, WV, 1>()};
  OCCP(2, 4, 2) OCCP(2, 4, 4) OCCP(2, 4, 8) OCCP(2, 2, 4) OCCP(2, 2, 8) OCCP(2, 8, 2) OCCP(2, 8, 4)
  OCCP(4, 4, 4) OCCP(1, 4, 4) OCCP(1, 8, 4) OCCP(1, 16, 4) OCCP(1, 16, 8)
  OCCW(1, 4, 1) OCCW(1, 4, 2) OCCW(1, 4, 4) OCCW(2, 2, 1) OCCW(2, 2, 2) OCCW(2, 2, 4)
  return v;
}

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  size_t a = 0, b;
  while ((b = s.find(c, a)) != std::string::npos) { out.push_back(s.substr(a, b - a)); a = b + 1; }
  if (a < s.size()) out.push_back(s.substr(a));
  return out;
}

int main(int argc, char** argv) {
  // shape -> calls per decode step (Qwen3.5-9B, from the steady-state trace)
  std::vector<std::tuple<int, int, int>> shapes = {
      {248320, 4096, 1}, {24576, 4096, 32}, {12288, 4096, 24},
      {10240, 4096, 8},  {4096, 12288, 32}, {4096, 4096, 32}};
  std::string vsel = "base,d1p0y2u2,d0p1y2u2,d1p1y2u2";
  int iters = 100, rounds = 3, rot_mb = 1536, cu = 0;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nxt = [&]() { return std::string(argv[++i]); };
    if (a == "--variants") vsel = nxt();
    else if (a == "--iters") iters = atoi(nxt().c_str());
    else if (a == "--rounds") rounds = atoi(nxt().c_str());
    else if (a == "--rot-mb") rot_mb = atoi(nxt().c_str());
    else if (a == "--cu") cu = atoi(nxt().c_str());
    else if (a == "--shapes") {
      shapes.clear();
      for (auto& s : split(nxt(), ',')) {
        auto p = split(s, 'x');
        shapes.emplace_back(atoi(p[0].c_str()), atoi(p[1].c_str()), p.size() > 2 ? atoi(p[2].c_str()) : 1);
      }
    } else if (a == "--list") {
      for (auto& kv : make_variants()) printf("%s\n", kv.first.c_str());
      return 0;
    } else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
  }
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  if (!cu) cu = prop.multiProcessorCount;
  printf("device %s (%s), CuCount=%d, rot %d MB, %d rounds x %d iters\n", prop.name,
         prop.gcnArchName, cu, rot_mb, rounds, iters);
  auto all = make_variants();
  std::vector<std::string> names = vsel == "all" ? std::vector<std::string>{} : split(vsel, ',');
  if (vsel == "all") for (auto& kv : all) names.push_back(kv.first);
  for (auto& n : names)
    if (!all.count(n)) { fprintf(stderr, "unknown variant %s (see --list)\n", n.c_str()); return 1; }

  std::map<std::string, double> step_us;
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
  printf("%-14s %-11s %9s %7s %9s %9s\n", "shape", "variant", "us", "GB/s", "vs base", "max_rel");
  for (auto [M, K, cnt] : shapes) {
    size_t n = (size_t)M * K, bytes = n * 2;
    int nb = std::max<int>(2, (int)((size_t)rot_mb * (1 << 20) / bytes) + 1);
    std::vector<bf16*> W(nb);
    for (int b = 0; b < nb; b++) {
      CHECK(hipMalloc(&W[b], bytes));
      fill_rand<<<1024, 256>>>(W[b], n, 1234 + b);
    }
    bf16 *x, *y; float* yref;
    CHECK(hipMalloc(&x, K * 2)); CHECK(hipMalloc(&y, M * 2)); CHECK(hipMalloc(&yref, M * 4));
    fill_rand<<<64, 256>>>(x, K, 99);
    ref_gemv<<<M, 256>>>(W[0], x, yref, M, K);
    std::vector<float> href(M);
    std::vector<bf16> hy(M);
    CHECK(hipMemcpy(href.data(), yref, M * 4, hipMemcpyDeviceToHost));
    double ref_scale = 0;
    for (float v : href) ref_scale = std::max(ref_scale, (double)fabs(v));

    std::map<std::string, std::vector<double>> t;
    for (auto& name : names) {  // correctness
      auto& [yt, launch] = all[name];
      if (M % yt) { printf("  %s: M %% YTILE != 0, skipped\n", name.c_str()); continue; }
      CHECK(hipMemset(y, 0xff, M * 2));
      launch(W[0], x, y, M, K, cu);
      CHECK(hipGetLastError());
      CHECK(hipMemcpy(hy.data(), y, M * 2, hipMemcpyDeviceToHost));
      double err = 0;
      for (int i = 0; i < M; i++) err = std::max(err, (double)fabs(__bfloat162float(hy[i]) - href[i]));
      t[name].push_back(err / ref_scale);
    }
    for (int r = 0; r < rounds; r++)  // interleave variants across rounds to cancel drift
      for (auto& name : names) {
        if (!t.count(name)) continue;
        auto& launch = all[name].second;
        int c = 0;
        for (int w = 0; w < 10; w++) launch(W[c++ % nb], x, y, M, K, cu);
        CHECK(hipEventRecord(e0));
        for (int it = 0; it < iters; it++) launch(W[c++ % nb], x, y, M, K, cu);
        CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
        float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
        t[name].push_back(ms * 1e3 / iters);
      }
    double base_us = 0;
    for (auto& name : names) {
      if (!t.count(name)) continue;
      std::vector<double> v(t[name].begin() + 1, t[name].end());
      std::sort(v.begin(), v.end());
      double us = v[v.size() / 2];
      if (!base_us) base_us = us;
      step_us[name] += us * cnt;
      char shp[32]; snprintf(shp, sizeof shp, "%dx%d", M, K);
      printf("%-14s %-11s %9.1f %7.0f %+8.1f%% %9.1e%s\n", shp, name.c_str(), us, bytes / us / 1e3,
             100 * (us - base_us) / base_us, t[name][0], t[name][0] > 1e-2 ? "  <-- WRONG?" : "");
    }
    for (auto p : W) CHECK(hipFree(p));
    CHECK(hipFree(x)); CHECK(hipFree(y)); CHECK(hipFree(yref));
  }
  printf("\nGEMV time per decode step (sum of us x calls/step over shapes above):\n");
  double b = step_us[names[0]];
  for (auto& name : names)
    printf("  %-11s %8.3f ms  (%+.2f%% vs %s)\n", name.c_str(), step_us[name] / 1e3,
           100 * (step_us[name] - b) / b, names[0].c_str());
  return 0;
}
