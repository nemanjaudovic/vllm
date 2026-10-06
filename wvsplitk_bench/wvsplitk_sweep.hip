// Config sweep for vllm's wvSplitK (bf16, any N=1..5) on gfx11/gfx12, using vllm's own kernels
// (vllm_kernels.inc, produced by extract_kernels.sh from csrc/rocm/skinny_gemms.cu).
//
// Each variant = (DOT, YTILE, UNRL, WVPRGRP). Kernel choice (sml / hf / big) follows vllm's
// WVSPLITK_CFG rule: N*K fits LDS and M%YTILE==0 -> sml; N*K <= 1.2*LDS -> hf; else big.
// "vllm" row = what vllm's current gfx1x heuristic picks for that shape (16 waves, fp32 dot).
//
//   ./build_sweep.sh <N>        -> ./wvsplitk_sweep_n<N>
//   ./wvsplitk_sweep_n<N> [--shapes MxK[xcalls],...] [--rounds 3] [--csv out.csv] [--cu C]
//                         [--only v1,v2] [--target-ms 20] [--dot-only]
#include "wv_common.h"

#ifndef SWEEP_N
  #define SWEEP_N 1
#endif

struct Variant {
  std::string name;
  int dot;  // 0 fp, 1 nd, 2 nz
  int yt, u, wv;
  Launcher launch;
};

template <int ND, int YT, int U>
void add_wv(std::vector<Variant>& v) {
  v.push_back({"", ND, YT, U, 2, make_launcher<ND, YT, U, 2, SWEEP_N>()});
  v.push_back({"", ND, YT, U, 4, make_launcher<ND, YT, U, 4, SWEEP_N>()});
  v.push_back({"", ND, YT, U, 8, make_launcher<ND, YT, U, 8, SWEEP_N>()});
  v.push_back({"", ND, YT, U, 16, make_launcher<ND, YT, U, 16, SWEEP_N>()});
}
template <int ND, int YT>
void add_u(std::vector<Variant>& v) {
  add_wv<ND, YT, 1>(v); add_wv<ND, YT, 2>(v); add_wv<ND, YT, 4>(v); add_wv<ND, YT, 8>(v);
}
template <int ND>
void add_yt(std::vector<Variant>& v) {
  add_u<ND, 1>(v); add_u<ND, 2>(v); add_u<ND, 3>(v); add_u<ND, 4>(v);
}
std::vector<Variant> make_variants() {
  std::vector<Variant> v;
  add_yt<0>(v);
  add_yt<1>(v);
  // dot mode 2 (nz: zero-acc native dot + add) is in wvsplitk_accuracy only: it is less accurate
  // than both fp and nd under cancellation (results/gfx1100_w7900/accuracy.csv), so not a candidate
  for (auto& x : v)
    x.name = std::string(DOT_PREFIX[x.dot]) + "_y" + std::to_string(x.yt) + "u" +
             std::to_string(x.u) + "w" + std::to_string(x.wv);
  return v;
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

int main(int argc, char** argv) {
  const int N = SWEEP_N;
  // MxK x calls-per-decode-step (Qwen3.5-9B; replace with your model's shapes)
  std::vector<std::tuple<int, int, int>> shapes = {
      {248320, 4096, 1}, {24576, 4096, 32}, {12288, 4096, 24},
      {10240, 4096, 8},  {4096, 12288, 32}, {4096, 4096, 32}};
  int rounds = 3, rot_mb = 1536, cu = 0;
  double target_ms = 20;  // timed-loop length per measurement
  std::string csv = "sweep_n" + std::to_string(N) + ".csv", only;
  bool dot_only = false;  // per shape: only vllm's current config, fp32 dot vs native dot
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nxt = [&]() { return std::string(argv[++i]); };
    if (a == "--rounds") rounds = atoi(nxt().c_str());
    else if (a == "--rot-mb") rot_mb = atoi(nxt().c_str());
    else if (a == "--cu") cu = atoi(nxt().c_str());
    else if (a == "--target-ms") target_ms = atof(nxt().c_str());
    else if (a == "--csv") csv = nxt();
    else if (a == "--only") only = nxt();  // comma list of variant names
    else if (a == "--dot-only") dot_only = true;
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
    std::string def = vllm_default(M, K, N, cu, std::string(prop.gcnArchName).find("gfx1151") == 0);

    // correctness + rough time per call (to size the timed loop)
    std::vector<double> err(vars.size(), -1), est(vars.size());
    for (size_t i = 0; i < vars.size(); i++) {
      if (dot_only && vars[i].name.substr(3) != def.substr(3)) continue;  // fp/nd/nz of the default
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
        int iters = std::clamp((int)(target_ms / std::max(est[i], 1e-3)), 10, 200);
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
    if (def_us > 0)
      printf("%7dx%-6d N=%d  vllm(%s) %8.1f us  best %-14s %8.1f us  %+5.1f%%  [kernel %s]\n", M,
             K, N, def.c_str(), def_us, best.c_str(), best_us, 100 * (best_us - def_us) / def_us,
             kernel_for(M, K, N, 2));
    else  // default config excluded by --only
      printf("%7dx%-6d N=%d  vllm(%s) not run          best %-14s %8.1f us  [kernel %s]\n", M, K,
             N, def.c_str(), best.c_str(), best_us, kernel_for(M, K, N, 2));
    fflush(f);
    for (auto p : W) CHECK(hipFree(p));
    CHECK(hipFree(x)); CHECK(hipFree(y)); CHECK(hipFree(yref));
  }
  fclose(f);
  return 0;
}
