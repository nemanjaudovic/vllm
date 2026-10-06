# Handoff: tune vLLM's wvSplitK skinny GEMV for gfx1201 (RDNA4)

You are continuing a kernel-tuning effort started on a Radeon Pro W7900 (gfx1100). Your job: run the
same **model-agnostic** config sweep on **gfx1201** with the harness in this directory, using the vllm
checkout the user gives you (branch `gfx11-wvsplitk-tuning` of github.com/nemanjaudovic/vllm), and turn
the results into per-N tile rules for gfx1201 (like gfx1151's in PR #40784). Start with N=1 only, report,
then do N=2..5. Do not change the harness methodology without telling the user;
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
| `wvsplitk_sweep.hip` | compiles them twice (`fp_` = current dot, `nd_` = native dot); grid YTILE {1,2,3,4} x UNRL {1,2,4,8} x WvPrGrp {2,4,8,16} = 128 variants; vllm's sml/hf/big dispatch; fp32 reference check; weights rotated over >= 1.5 GB so every call reads DRAM; median of interleaved rounds. Flags: `--shapes MxK[xcalls],...`, `--only v1,v2`, `--rounds`, `--target-ms`, `--csv`, `--cu` |
| `build_sweep.sh N [arch]` | builds `wvsplitk_sweep_n<N>`, writes `isa_sweep_n<N>/resources.txt` (VGPR/scratch per instantiation). gfx1100 box: ~80 s for N=1, ~15 min link for N=5 (spilling configs) |
| `general_sweep.sh [arch] [Ns] [rounds] [target_ms]` | **main entry point**: builds the given N in parallel, runs each on the 117-shape grid in `general_shapes.txt`, logs to `general_sweep_<arch>.log`, writes `sweep_<arch>_general_n<N>.csv` |
| `general_shapes.txt` | M x K grid: K {1024,1536,2048,2560,3072,4096,5120,6144,8192,12288,14336,16384} x M {1024,2048,4096,6144,8192,12288,16384,28672,65536,151936}, weights capped at 2.6 GB |
| `rule_explorer.py` | **main analysis**: every shape weighted equally, configs ranked by geomean slowdown vs the per-shape best. Shows the vllm default vs best single config vs rule families (per K, per (K, M class), per (kernel, M class)) vs the oracle, a robust shortlist, and with `--maps` K x M winner/loss maps |
| `sweep_report.py` | per-model view: shapes weighted by calls per decode step (default shapes = Qwen3.5-9B) |
| `wvsplitk_bench.hip`, `build.sh` | older N=1 experiments (pipelined loads, WGs per WGP); not needed |
| `results/gfx1100_w7900/` | reference: `general_n<N>.csv` (general grid), `qwen_n<N>.csv` (Qwen3.5-9B shapes) |

Variant names: `nd_y4u4w2` = native dot, YTILE 4, UNRL 4, 2 waves per WG.

## 5. What to run on gfx1201

0. Environment, recorded in your report: `rocminfo | grep -m1 gfx`, GPU name, the CuCount the
   harness prints, ROCm/hipcc version, free VRAM (needs ~6 GB). Make sure the GPU is idle:
   `amd-smi metric -u` (GFX activity ~0) and no other jobs. Background load moved results ~1% on gfx1100.
   If `hipcc` is not on PATH: `export HIPCC=/path/to/hipcc`.
1. N=1 general sweep (gfx1100 took ~10 min):
   ```bash
   cd wvsplitk_bench
   ./general_sweep.sh gfx1201 1
   python3 rule_explorer.py sweep_gfx1201_general_n1.csv --maps
   ```
   Then check the native dot on gfx12: grep `isa_sweep_n1/*.s` for `v_dot2_f32_bf16`. If it is
   missing, the `nd_` variants aren't what they claim to be. Stop and report.
   Any `WRONG RESULT` in `general_sweep_gfx1201.log` is a correctness bug. Report it, never ignore it.
   **Stop here and report N=1 to the user** (see section 6) before continuing.
2. N=2..5: `./general_sweep.sh gfx1201 "2 3 4 5"` (~1 h+), then
   `python3 rule_explorer.py sweep_gfx1201_general_n*.csv --maps`.
3. Optional, Qwen3.5-9B view (the user's real workload): `./wvsplitk_sweep_n$N --rounds 3 --csv
   sweep_gfx1201_qwen_n$N.csv` for each built N, then `python3 sweep_report.py sweep_gfx1201_qwen_n*.csv`.
4. Copy finished CSVs to `results/gfx1201_<card>/` as `general_n<N>.csv` / `qwen_n<N>.csv` and
   commit. Ask the user before pushing.

## 6. What to report back (compare with gfx1100 in section 8)

- Per N: rule_explorer's summary block (default / single / rule families / oracle) and the shortlist.
- Does "fewer waves + deeper loads" hold on gfx1201? Does the native dot help?
- Which features change the winner: K (look at the power-of-2 vs K%1024==512 rows), M (few rounds
  vs many), N, and the sml/hf/big kernel. These become rule conditions.
- A draft `else if (on_gfx12())` branch for `WVSPLIT_TILE` in gfx1151 style: few rules, each
  justified by the maps. Report its geomean vs the oracle and vs the default. Prefer 3-6 rules over
  a per-shape table; on gfx1100 one config per K (6 configs) already reached 1.004 vs oracle.

## 7. Known gaps / open questions

- The config grid is complete relative to gfx1151's rules: their configs are all subsets of
  YTILE {1,2,4} x UNRL {1,2,4} at 16 waves. Not swept: grid size (WGs per WGP; no gain on gfx1100)
  and LDS size (fixed 64 KB, hardware max per WG).
- N=4-5 at large K use `wvSplitK_hf_big_`, which re-stages x through LDS with barriers on every row
  round. Hypothesis: the `hf_` path (tail of x from L2) would be faster. A `--force-kernel` option
  to test this is not implemented yet.
- Nothing has been ported into vllm's dispatch yet, and there is no end-to-end number yet for any
  arch. Expected on gfx1100: ~+2-3% output tok/s for the single-user Qwen benchmark.

## 8. gfx1100 (W7900, CuCount 48) reference results

N=1 general grid (117 shapes; geomean slowdown vs per-shape best):

| | geomean | worst shape |
|---|---|---|
| vllm default (`fp_y2u2w16` etc.) | 1.040 | 1.203 (K=3072, M=1024) |
| best single config `nd_y1u2w8` | 1.009 | 1.049 |
| one config per K (6 distinct) | 1.004 | 1.033 |
| one config per (K, M class) (17 distinct) | 1.001 | 1.007 |

- The native dot (`nd_`) wins almost every shape.
- The default is within ~2% of best only at K=1536 and 2560 (K%1024==512, the
  only K whose row stride is not a multiple of 2 KB). At power-of-2-ish K it loses 4-20%, worst at
  small M. Hypothesis: many concurrent row streams with power-of-2 strides collide on DRAM channels.
  Fewer/deeper streams avoid it. Unverified with counters. Check whether gfx1201 shows the same K
  pattern.
- Large M (many row rounds) prefers YTILE 3-4 with 2 waves (`nd_y4u4w2`, `nd_y3u8w2`); small M
  prefers YTILE 1 with 8-16 waves (`nd_y1u8w16`).
- Qwen3.5-9B weighted (sweep_report.py): N=1 `nd_y4u4w2` -3.4%, N=2 `nd_y4u4w2` -4.7%,
  N=3 `nd_y2u4w4` -3.8% GEMV time vs default.
- General grid, all N (geomean vs oracle; `results/gfx1100_w7900/general_n<N>.csv`):

  | N | vllm default (worst) | best single config | one config per K | per (K, M class) |
  |---|---|---|---|---|
  | 1 | 1.040 (1.20) | 1.009 `nd_y1u2w8` | 1.004 (6 cfgs) | 1.001 (17) |
  | 2 | 1.076 (1.41) | 1.010 `nd_y1u2w8` | 1.005 (7) | 1.001 (14) |
  | 3 | 1.074 (1.41) | 1.010 `nd_y1u4w16` | 1.005 (6) | 1.001 (15) |
  | 4 | 1.093 (1.45) | 1.009 `nd_y1u4w8` | 1.005 (5) | 1.001 (12) |
  | 5 | 1.081 (1.43) | 1.010 `nd_y1u4w8` | 1.006 (5) | 1.001 (13) |

- Native dot vs fp32 dot at the same config (geomean over shapes): N=1: -1.3% at 16 waves,
  -16% at 2 waves; N=4: -9% at 16 waves, -27% at 2 waves. Low-wave configs are only competitive
  with the native dot. Best-nd vs best-fp per shape: -0.4% (N=1) to -1.2% (N=4).
  Max relative error vs the fp32 reference is identical for nd and fp (bitwise diff not measured).
