# Handoff: tune vLLM's wvSplitK skinny GEMV for gfx1201 (RDNA4)

You are continuing a kernel-tuning effort started on a Radeon Pro W7900 (gfx1100). Your job: run the
same config sweep on **gfx1201** with the harness in this directory, using the vllm checkout the user
gives you (branch `gfx11-wvsplitk-tuning` of github.com/nemanjaudovic/vllm), and turn the results into
per-N tile rules for gfx1201. Do not change the harness methodology without telling the user;
gfx1100 and gfx1201 results must stay comparable.

## 1. Background

- Workload being optimized: `vllm bench throughput --model Qwen/Qwen3.5-9B ... --max-num-seqs 1`
  (single-user decode, bf16). On W7900, ~93% of GPU time is one kernel: `wvSplitK_hf_sml_`
  (`csrc/rocm/skinny_gemms.cu`), a GEMV `y[n][m] = sum_k W[m][k] * x[n][k]` for N = 1..5 tokens.
- It is purely DRAM-bound (~1 FLOP/byte). The question is never TFLOPS; it is GB/s against what a
  pure-read kernel achieves (W7900: ~766 GB/s read roof, 89% of 864 GB/s spec).
- Result on gfx1100: changing only the launch config (fewer waves per workgroup, deeper loads per
  wave) plus a native bf16 dot instruction makes all GEMVs 3.4-4.7% faster. More waves is slower.

## 2. How the kernel works (read the source too: `wvSplitK_hf_sml_`, ~L366)

```
grid  = CuCount workgroups (hipDeviceProp.multiProcessorCount, = vllm num_compute_units()),
        persistent, 1 WG per WGP
block = 32 lanes x WvPrGrp waves
1. whole WG copies x (N x K bf16) into one shared 64 KB LDS buffer; __syncthreads
2. each wave owns YTILE consecutive rows of W. Loop over K in steps of 32*8*UNRL elements:
   each lane loads YTILE*UNRL 16-byte chunks of W (nontemporal; the only DRAM traffic),
   reads N*UNRL 16-byte chunks of x from LDS, fp32-accumulates with DOT2C
3. reduce 32 lane partials (DPP + shfl_xor 16); lane 31 writes YTILE outputs per token
4. m += CuCount*WvPrGrp*YTILE; repeat until m >= M
```
- `max_lds_len = LDS_SIZE/2` is an element count (bf16 = 2 bytes), not a per-wave split.
- Occupancy = WvPrGrp/4 waves per SIMD (4 SIMD32 per WGP; LDS is a fixed 64 KB, so exactly 1 WG
  per WGP). VGPRs ~ 4*YTILE*UNRL + 4*N*UNRL + N*YTILE + ~20. They never limit occupancy here;
  only avoid going past 256 (spills; see `resources.txt`).
- `mindiv(M, CuCount*YTILE, WvPrGrp)` picks the fewest active waves that need the same number of
  row rounds (load balance). **Bug in vllm:** it always tries 13 candidates, so WvPrGrp < 13
  divides by <= 0. The harness uses a guarded copy; any vllm port must fix it.
- DOT2C for bf16 on gfx1x unpacks to fp32 (mul + add). It was inherited from the gfx9 path
  (PR #15830, then the gfx1x port PR #34709 switched only fp16 to the native dot). The native
  `__builtin_amdgcn_fdot2_f32_bf16` (`v_dot2_f32_bf16`) compiles for gfx1100 and gfx1201.
  Accumulation stays fp32; error vs the fp32 reference is unchanged.

## 3. When vllm uses which kernel (dimension conditions)

`vllm/model_executor/layers/utils.py::rocm_unquantized_gemm_impl`. Python names:
n = tokens, m = out_features, k = in_features.
- `use_skinny`: `VLLM_ROCM_USE_SKINNY_GEMM` (default on), gfx9 or gfx1x, fp16/bf16, `k % 8 == 0`,
  contiguous weight/bias. Then `(m == 1 or m > 8) and 0 < n <= 5` -> `ops.wvSplitK`. n > 5 goes
  to hipBLASLt (or aiter tgemm).
- Inside `wvSplitK` (C++), with L = 32768 elements:
  - `N*K <= L and M % YTILE == 0` -> `wvSplitK_hf_sml_` (x fully in LDS)
  - `N*K <= 1.2*L` -> `wvSplitK_hf_` (the tail of x read from global)
  - else -> `wvSplitK_hf_big_` (x staged in chunks of L/N with barriers, re-staged every row round)
- Tile choice: `WVSPLIT_TILE`. gfx1151 has its own rule list (PR #40784, the model for what we
  want). Every other gfx1x, **including gfx1201**, uses the generic `sYT = ceil(M/(CuCount*4))`
  heuristic with WvPrGrp=16 and the fp32 bf16 dot. The harness's "vllm default" row reproduces it.

## 4. Harness (this directory)

| file | purpose |
|---|---|
| `extract_kernels.sh` | copies vllm's real kernels (sml/hf/big, min__, mindiv) out of `../csrc/rocm/skinny_gemms.cu` into `vllm_kernels.inc`, minus DOT2C |
| `wvsplitk_sweep.hip` | compiles them twice (`fp_` = current dot, `nd_` = native dot); grid YTILE {1,2,3,4} x UNRL {1,2,4,8} x WvPrGrp {2,4,8,16} = 128 variants; vllm's sml/hf/big dispatch; fp32 reference check; weights rotated over >= 1.5 GB so every call reads DRAM; median of interleaved rounds |
| `build_sweep.sh N [arch]` | builds `wvsplitk_sweep_n<N>` (~80 s on the gfx1100 box), writes `isa_sweep_n<N>/resources.txt` (VGPR/scratch per instantiation) |
| `sweep_report.py` | per N: vllm default vs best single config vs best per shape, plus the top configs |
| `wvsplitk_bench.hip`, `build.sh` | older N=1 experiments (pipelined loads, WGs per WGP); not needed for the sweep |
| `results/gfx1100_w7900/` | reference CSVs, N=1..3 |

Variant names: `nd_y4u4w2` = native dot, YTILE 4, UNRL 4, 2 waves per WG.

## 5. What to run on gfx1201

0. Check the environment and record it in your report: `rocminfo | grep -m1 gfx`, GPU name, the
   CuCount the harness prints, ROCm/hipcc version. Make sure the GPU is idle:
   `amd-smi metric -u` (GFX activity ~0) and no other jobs. Background load moved results ~1% on gfx1100.
   If `hipcc` is not on PATH: `export HIPCC=/path/to/hipcc`.
1. Sanity check the native dot on gfx12: after building, grep the generated ISA in
   `isa_sweep_n1/*.s` for `v_dot2_f32_bf16`. If it is missing, the `nd_` variants are not what
   they claim to be. Stop and report.
2. Primary sweep (Qwen3.5-9B shapes, weighted by calls per decode step; this is the default set):
   ```bash
   cd wvsplitk_bench
   for N in 1 2 3 4 5; do
     ./build_sweep.sh $N gfx1201 && ./wvsplitk_sweep_n$N --rounds 3 --csv sweep_gfx1201_n$N.csv
   done
   python3 sweep_report.py sweep_gfx1201_n*.csv --top 8
   ```
   Any `WRONG RESULT` line means a correctness bug. Report it; never ignore it.
3. Generalization sweep. Rules must not be fit to 6 shapes. Run every N on a broader shape set
   (representative decode shapes of Llama-3.1-8B, Qwen3-14B, Qwen3-1.7B; MxK):
   ```bash
   S=6144x4096,4096x4096,28672x4096,4096x14336,128256x4096,7168x5120,5120x5120,34816x5120,5120x17408,151936x5120,4096x2048,2048x2048,12288x2048,2048x6144,151936x2048
   ./wvsplitk_sweep_n$N --shapes $S --rounds 3 --csv sweep_gfx1201_broad_n$N.csv
   python3 sweep_report.py sweep_gfx1201_broad_n*.csv --unweighted
   ```
4. Copy the finished CSVs to `results/gfx1201_<card>/` and commit them (ask the user before pushing).
5. Optional, gives a roof to compare against: if Triton is available, the pure-read roof script
   from the gfx1100 box (`read_roof.py`; ask the user for it) at the same byte sizes.

## 6. What to report back

- Per N: vllm default, best single config, and best per shape (ms and %), and whether the winners
  match gfx1100's (N=1 `nd_y4u4w2` -3.4%; N=2 `nd_y4u4w2` -4.7%; N=3 `nd_y2u4w4` -3.8%).
- Does "fewer waves + deeper loads" hold on gfx1201? Does the native dot help?
- Which shape features (K, M, N, the sml/hf/big kernel) change the winner. These become rule
  conditions.
- A draft `else if (on_gfx12())` branch for `WVSPLIT_TILE` in gfx1151 style: few rules, each
  justified by the sweep, with the gain of the rules vs best-per-shape. Prefer simple rules;
  per-shape tuning was only 0.3-0.9% better than a single config on gfx1100.

## 7. Known gaps / open questions

- The config grid is complete relative to gfx1151's rules: their configs are all subsets of
  YTILE {1,2,4} x UNRL {1,2,4} at 16 waves. Not swept: grid size (WGs per WGP; no gain on gfx1100)
  and LDS size (fixed 64 KB, hardware max per WG).
- N=4-5 at K=12288 use `wvSplitK_hf_big_`, which re-stages x through LDS with barriers on every row
  round. Hypothesis: the `hf_` path (tail of x from L2) would be faster. A `--force-kernel` option
  to test this is not implemented yet.
- Hypothesis for why fewer waves win: fewer concurrent weight-row streams means better DRAM page
  and channel locality. Not verified with counters.
- Nothing has been ported into vllm's dispatch yet, and there is no end-to-end number yet for any
  arch. Expected on gfx1100: ~+2-3% output tok/s for the single-user benchmark.
