#!/usr/bin/env python3
"""Tuning with vs without weight row-stride padding (vllm PR #55090), from two general sweeps.

  python3 pad_compare.py --nopad results/gfx1100_w7900/general_n{1..5}.csv \\
                         --pad   results/gfx1100_w7900/general_pad_n{1..5}.csv

All numbers are geomean time over shapes relative to TODAY's vllm (unpadded, current config) = 1.000,
every shape weighted equally. Shapes whose K is not padded by the PR (K*2 % 2048 != 0) are identical
in both modes except for noise; the 'padded shapes' columns restrict to the ones the PR would pad.
"""
import argparse, collections, csv, math

ap = argparse.ArgumentParser()
ap.add_argument("--nopad", nargs="+", required=True)
ap.add_argument("--pad", nargs="+", required=True)
a = ap.parse_args()
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))


def load(files):
    t = collections.defaultdict(dict)  # (N, M, K) -> variant -> us
    d = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            s = (int(r["N"]), int(r["M"]), int(r["K"]))
            if float(r["max_rel_err"]) < 1e-2:
                t[s][r["variant"]] = float(r["us"])
            if r["is_vllm_default"] == "1":
                d[s] = r["variant"]
    return t, d


tn, dn = load(a.nopad)
tp, dp = load(a.pad)
Ns = sorted({s[0] for s in tn} & {s[0] for s in tp})
hdr = (f"{'N':>2} {'shapes':>6} | {'pad, today cfg':>14} {'tuned 1 cfg':>11} {'pad+tuned 1':>11} | "
       f"{'oracle':>7} {'pad oracle':>10} {'both':>7} | {'pad wins>1%':>11} {'nopad wins>1%':>13}")
for subset in ("all shapes", "padded shapes (K%1024==0)", "padded, M<=8192", "padded, M>8192"):
    print(f"\n== {subset}  (geomean time vs today's vllm = 1.000)\n{hdr}")
    for N in Ns:
        shapes = [s for s in tn if s[0] == N and s in tp and s in dn]
        if subset != "all shapes":
            shapes = [s for s in shapes if (s[2] * 2) % 2048 == 0]
        if subset.endswith("M<=8192"):
            shapes = [s for s in shapes if s[1] <= 8192]
        if subset.endswith("M>8192"):
            shapes = [s for s in shapes if s[1] > 8192]
        if not shapes:
            continue
        base = {s: tn[s][dn[s]] for s in shapes}
        cfgs = set.intersection(*(set(tn[s]) & set(tp[s]) for s in shapes))
        rel = lambda t, v: gm([t[s][v] / base[s] for s in shapes])
        best_np = min(cfgs, key=lambda v: rel(tn, v))
        best_p = min(cfgs, key=lambda v: rel(tp, v))
        orc_np = gm([min(tn[s].values()) / base[s] for s in shapes])
        orc_p = gm([min(tp[s].values()) / base[s] for s in shapes])
        both = gm([min(min(tn[s].values()), min(tp[s].values())) / base[s] for s in shapes])
        pw = sum(min(tp[s].values()) < 0.99 * min(tn[s].values()) for s in shapes)
        nw = sum(min(tn[s].values()) < 0.99 * min(tp[s].values()) for s in shapes)
        print(f"{N:>2} {len(shapes):>6} | {gm([tp[s][dp[s]] / base[s] for s in shapes]):14.4f} "
              f"{rel(tn, best_np):11.4f} {rel(tp, best_p):11.4f} | {orc_np:7.4f} {orc_p:10.4f} {both:7.4f} | "
              f"{pw:>11} {nw:>13}")
        if subset == "all shapes":
            print(f"{'':>10}  best single config: no pad {best_np}, pad {best_p}")
