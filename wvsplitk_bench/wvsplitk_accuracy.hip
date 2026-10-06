// Numerical comparison of wvSplitK's bf16 dot: current fp32 path (fp) vs native v_dot2_f32_bf16 with the
// accumulator as 3rd operand (nd) vs native dot with zero accumulator + separate add (nz), against fp64, at vllm's current tile config for each shape, N = 1..5.
//
//   ./build_accuracy.sh && ./wvsplitk_accuracy [--weights file.bin:M:K[:name],...] [--csv out.csv]
//
// Per (data, shape, N) it reports, for fp and nd:
//   rel   max |y - ref| / max |ref|
//   ulp   mean |y - ref| in bf16 ulps of ref (|ref| > 0 only)
// and between fp and nd outputs: % bitwise equal, max bf16-ulp distance, % where nd is closer to
// fp64 / % where fp is closer, NaN/Inf pattern mismatches vs the reference, and whether an alternate
// tile config gives bitwise identical output (summation order per lane doesn't depend on the tile).
// Data sets: normal | outlier (1% activation channels x50) | tiny (fp32-denormal products) |
//            huge (near bf16 max, overflow) | special (inf/nan in x) | cancel (paired +-w) | real weights
#include "wv_common.h"
#include <cstdint>
#include <cstring>
#include <map>
#include <random>

using Key = std::tuple<int, int, int, int>;  // N, YTILE, UNRL, dot mode (0 fp, 1 nd, 2 nz)
std::map<Key, Launcher> L;

template <int N, int YT, int U>
void add() {
  L[{N, YT, U, 0}] = make_launcher<0, YT, U, 16, N>();
  L[{N, YT, U, 1}] = make_launcher<1, YT, U, 16, N>();
  L[{N, YT, U, 2}] = make_launcher<2, YT, U, 16, N>();
}
template <int N>
void add_n() {  // every (YTILE, UNRL) any vllm gfx1x rule picks today, at 16 waves
  add<N, 1, 1>(); add<N, 1, 2>(); add<N, 1, 4>(); add<N, 2, 2>();
  add<N, 2, 4>(); add<N, 3, 2>(); add<N, 4, 1>(); add<N, 4, 2>();
}

__global__ void ref_gemv_f64(const bf16* W, const bf16* x, double* y, int M, int K) {
  __shared__ double red[256];
  int row = blockIdx.x, n = blockIdx.y;
  double acc = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    acc += (double)__bfloat162float(W[(size_t)row * K + k]) * (double)__bfloat162float(x[(size_t)n * K + k]);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) y[(size_t)n * M + row] = red[0];
}

uint16_t bits(bf16 v) { uint16_t b; memcpy(&b, &v, 2); return b; }
float tof(bf16 v) { return __bfloat162float(v); }
int64_t ordered(uint16_t b) { return (b & 0x8000) ? -(int64_t)(b & 0x7fff) : (int64_t)b; }
double bf16_ulp(double r) {  // spacing of bf16 numbers at |r| (8 significant bits)
  int e; std::frexp(std::fabs(r), &e);
  return std::ldexp(1.0, std::max(e - 8, -133));
}
bf16 b(float f) { return __float2bfloat16(f); }

void make_data(const std::string& mode, std::vector<bf16>& W, std::vector<bf16>& x, int M, int K, int N,
               std::mt19937& g) {
  std::normal_distribution<float> nd(0.f, 1.f);
  if (W.empty()) {  // synthetic weights
    W.resize((size_t)M * K);
    if (mode == "cancel") {  // w[k+1] = -w[k] (+ tiny noise): sums cancel, rounding of adds dominates
      for (size_t i = 0; i < W.size(); i += 2) {
        float v = 0.02f * nd(g);
        W[i] = b(v);
        if (i + 1 < W.size()) W[i + 1] = b(-v + 1e-4f * nd(g));
      }
    } else {
      float s = mode == "tiny" ? 1e-20f : mode == "huge" ? 1e18f : 0.02f;
      for (auto& w : W) w = b(s * nd(g));
    }
  }
  x.resize((size_t)N * K);
  float s = mode == "tiny" ? 1e-20f : mode == "huge" ? 1e18f : 1.f;
  for (auto& v : x) v = b(s * nd(g));
  if (mode == "outlier" || mode == "real")
    for (int k = 0; k < K; k += 100)  // 1% outlier channels, like LLM activations
      for (int n = 0; n < N; n++) x[(size_t)n * K + k] = b(50.f * nd(g));
  if (mode == "cancel")
    for (int k = 0; k + 1 < K; k += 2)
      for (int n = 0; n < N; n++) x[(size_t)n * K + k + 1] = x[(size_t)n * K + k];
  if (mode == "special") {
    float inf = INFINITY, nan = NAN;
    x[7] = b(inf);                       // token 0: +inf
    if (N > 1) x[K + 11] = b(nan);       // token 1: nan
    if (N > 2) { x[2 * K + 3] = b(inf); x[2 * K + 5] = b(-inf); }  // token 2: inf - inf
  }
}

struct Res {
  double rel[3], ulp[3];                  // per dot mode vs fp64
  double eq[3], maxulp[3], better[3], worse[3];  // per dot mode vs the fp path (index 0 unused)
  int special_mismatch[3];
  bool cfg_invariant[3];
};

int main(int argc, char** argv) {
  add_n<1>(); add_n<2>(); add_n<3>(); add_n<4>(); add_n<5>();
  std::vector<std::tuple<std::string, int, int, std::string>> real;  // file, M, K, name
  std::string csv = "accuracy.csv";
  std::vector<std::pair<int, int>> shapes = {{4096, 4096}, {4096, 12288}, {12288, 4096}, {24576, 4096},
                                             {2048, 2048}, {6144, 2560}, {248320, 4096}};
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--csv") csv = argv[++i];
    else if (a == "--shapes") {
      shapes.clear();
      for (auto& s : split(argv[++i], ',')) { auto p = split(s, 'x'); shapes.push_back({atoi(p[0].c_str()), atoi(p[1].c_str())}); }
    } else if (a == "--weights") {
      for (auto& s : split(argv[++i], ',')) {
        auto p = split(s, ':');
        real.emplace_back(p[0], atoi(p[1].c_str()), atoi(p[2].c_str()), p.size() > 3 ? p[3] : p[0]);
      }
    } else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
  }
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int cu = prop.multiProcessorCount;
  bool is1151 = std::string(prop.gcnArchName).find("gfx1151") == 0;
  printf("device %s (%s), CuCount=%d\n", prop.name, prop.gcnArchName, cu);
  FILE* f = fopen(csv.c_str(), "w");
  fprintf(f, "arch,data,M,K,N,config");
  for (const char* d : DOT_PREFIX) fprintf(f, ",rel_%s,ulp_%s,special_mismatch_%s,cfg_invariant_%s", d, d, d, d);
  for (const char* d : {"nd", "nz"})
    fprintf(f, ",pct_bitwise_equal_%s,max_ulp_diff_%s,pct_%s_closer,pct_fp_closer_vs_%s", d, d, d, d);
  fprintf(f, "\n");
  printf("%-17s %-13s %1s %-7s | %7s %7s %7s | %8s %5s %6s %6s | %8s %5s %6s %6s | %5s %3s\n", "data", "shape",
         "N", "config", "ulp fp", "ulp nd", "ulp nz", "nd %bit=", "maxU", "%ndBt", "%fpBt", "nz %bit=", "maxU",
         "%nzBt", "%fpBt", "spcl", "inv");

  std::vector<std::tuple<std::string, int, int, std::string>> jobs;  // data, M, K, weights file
  for (std::string mode : {"normal", "outlier", "cancel", "tiny", "huge", "special"})
    for (auto [M, K] : shapes)
      if (!((mode == "tiny" || mode == "huge" || mode == "special" || mode == "cancel") && (size_t)M * K > 64u << 20))
        jobs.emplace_back(mode, M, K, "");
  for (auto& [file, M, K, name] : real) jobs.emplace_back("real:" + name, M, K, file);

  std::mt19937 g(42);
  for (auto& [data, M, K, file] : jobs) {
    std::vector<bf16> hW;
    if (!file.empty()) {
      hW.resize((size_t)M * K);
      FILE* wf = fopen(file.c_str(), "rb");
      if (!wf || fread(hW.data(), 2, hW.size(), wf) != hW.size()) { fprintf(stderr, "cannot read %s\n", file.c_str()); return 1; }
      fclose(wf);
    }
    std::string mode = file.empty() ? data : "real";
    bf16 *dW, *dx, *dy; double* dref;
    CHECK(hipMalloc(&dW, (size_t)M * K * 2));
    CHECK(hipMalloc(&dx, (size_t)5 * K * 2)); CHECK(hipMalloc(&dy, (size_t)5 * M * 2));
    CHECK(hipMalloc(&dref, (size_t)5 * M * 8));
    for (int N = 1; N <= 5; N++) {
      std::vector<bf16> x;
      make_data(mode, hW, x, M, K, N, g);  // generates hW (synthetic) only if empty: once per shape
      if (N == 1) CHECK(hipMemcpy(dW, hW.data(), hW.size() * 2, hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
      ref_gemv_f64<<<dim3(M, N), 256>>>(dW, dx, dref, M, K);
      std::vector<double> ref((size_t)N * M);
      CHECK(hipMemcpy(ref.data(), dref, ref.size() * 8, hipMemcpyDeviceToHost));

      std::string def = vllm_default(M, K, N, cu, is1151);  // "fp_y<Y>u<U>w16"
      int yt = def[4] - '0', u = def[6] - '0';
      int ayt = yt == 1 ? 2 : 1, au = ayt == 1 ? 4 : 2;  // alternate config for the invariance check
      if (M % ayt) ayt = 1, au = 4;
      auto run = [&](int dot, int Y, int U) {
        CHECK(hipMemset(dy, 0, (size_t)N * M * 2));
        L.at({N, Y, U, dot})(dW, dx, dy, M, K, cu);
        CHECK(hipGetLastError());
        std::vector<bf16> h((size_t)N * M);
        CHECK(hipMemcpy(h.data(), dy, h.size() * 2, hipMemcpyDeviceToHost));
        return h;
      };
      std::vector<bf16> y[3] = {run(0, yt, u), run(1, yt, u), run(2, yt, u)};
      Res r{};
      double scale = 0;
      for (double v : ref) if (std::isfinite(v)) scale = std::max(scale, std::fabs(v));
      size_t cnt = ref.size(), nfin[3] = {}, eq[3] = {}, better[3] = {}, worse[3] = {};
      double ulpsum[3] = {};
      for (size_t i = 0; i < ref.size(); i++) {
        double rv = ref[i];
        bool special_ref = !std::isfinite(rv) || std::fabs(rv) > 3.39e38;  // inf/nan or bf16 overflow
        float v[3];
        for (int d = 0; d < 3; d++) {
          v[d] = tof(y[d][i]);
          bool special_y = !std::isfinite(v[d]);
          if (special_ref || special_y) {
            if (special_ref != special_y || (std::isnan(rv) != std::isnan(v[d]))) r.special_mismatch[d]++;
            continue;
          }
          double e = std::fabs(v[d] - rv);
          r.rel[d] = std::max(r.rel[d], scale > 0 ? e / scale : e);
          if (rv != 0) { ulpsum[d] += e / bf16_ulp(rv); nfin[d]++; }
        }
        for (int d = 1; d < 3; d++) {  // vs the fp path
          if (bits(y[0][i]) == bits(y[d][i])) { eq[d]++; continue; }
          if (!std::isfinite(v[0]) || !std::isfinite(v[d])) continue;
          r.maxulp[d] = std::max(r.maxulp[d], (double)std::llabs(ordered(bits(y[0][i])) - ordered(bits(y[d][i]))));
          double e0 = std::fabs(v[0] - rv), e1 = std::fabs(v[d] - rv);
          if (e1 < e0) better[d]++; else if (e0 < e1) worse[d]++;
        }
      }
      for (int d = 0; d < 3; d++) {
        r.ulp[d] = nfin[d] ? ulpsum[d] / nfin[d] : 0;
        auto alt = run(d, ayt, au);
        r.cfg_invariant[d] = memcmp(alt.data(), y[d].data(), alt.size() * 2) == 0;
        r.eq[d] = 100.0 * eq[d] / cnt; r.better[d] = 100.0 * better[d] / cnt; r.worse[d] = 100.0 * worse[d] / cnt;
      }
      char shp[32]; snprintf(shp, sizeof shp, "%dx%d", M, K);
      printf("%-17s %-13s %1d %-7s | %7.3f %7.3f %7.3f | %7.2f%% %5.0f %5.2f%% %5.2f%% | %7.2f%% %5.0f %5.2f%% %5.2f%% |"
             " %d/%d/%d %c%c%c\n",
             data.substr(0, 17).c_str(), shp, N, def.substr(3, 4).c_str(), r.ulp[0], r.ulp[1], r.ulp[2],
             r.eq[1], r.maxulp[1], r.better[1], r.worse[1], r.eq[2], r.maxulp[2], r.better[2], r.worse[2],
             r.special_mismatch[0], r.special_mismatch[1], r.special_mismatch[2],
             r.cfg_invariant[0] ? 'y' : 'N', r.cfg_invariant[1] ? 'y' : 'N', r.cfg_invariant[2] ? 'y' : 'N');
      fprintf(f, "%s,%s,%d,%d,%d,%s", prop.gcnArchName, data.c_str(), M, K, N, def.c_str());
      for (int d = 0; d < 3; d++)
        fprintf(f, ",%.3e,%.4f,%d,%d", r.rel[d], r.ulp[d], r.special_mismatch[d], (int)r.cfg_invariant[d]);
      for (int d = 1; d < 3; d++) fprintf(f, ",%.3f,%.0f,%.3f,%.3f", r.eq[d], r.maxulp[d], r.better[d], r.worse[d]);
      fprintf(f, "\n");
      fflush(f);
    }
    CHECK(hipFree(dW)); CHECK(hipFree(dx)); CHECK(hipFree(dy)); CHECK(hipFree(dref));
  }
  fclose(f);
  printf("-> %s\n", csv.c_str());
  return 0;
}
