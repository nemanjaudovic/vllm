#!/usr/bin/env python3
"""Summarize wvsplitk_sweep CSVs: per N, the best single config vs vllm's current default,
best per shape, and the top-k configs (weighted by calls/step).

  python3 sweep_report.py sweep_n1.csv sweep_n2.csv ... [--top 8] [--unweighted]
--unweighted: every shape counts once (use when calls/step don't reflect your target mix).
"""
import argparse, collections, csv

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--unweighted", action="store_true")
a = ap.parse_args()

rows = [r for f in a.csv for r in csv.DictReader(open(f))]
byN = collections.defaultdict(list)
for r in rows:
    byN[(r["arch"], int(r["N"]))].append(r)

for (arch, N), rs in sorted(byN.items()):
    shapes = {}  # (M,K) -> calls
    t = collections.defaultdict(dict)  # variant -> (M,K) -> us
    default, kern = {}, {}
    for r in rs:
        mk = (int(r["M"]), int(r["K"]))
        shapes[mk] = 1 if a.unweighted else int(r["calls"])
        if float(r["max_rel_err"]) < 1e-2:
            t[r["variant"]][mk] = float(r["us"])
        if r["is_vllm_default"] == "1":
            default[mk] = r["variant"]
        kern[(r["variant"], mk)] = r["kernel"]
    missing = [mk for mk in shapes if default.get(mk) not in t or mk not in t[default[mk]]]
    if missing:
        print(f"\n=== {arch}  N={N}: vllm default config missing for {missing} (run without --only); skipped")
        continue
    complete = {v: d for v, d in t.items() if len(d) == len(shapes)}
    tot = lambda v: sum(complete[v][mk] * c for mk, c in shapes.items())
    dtot = sum(t[default[mk]][mk] * c for mk, c in shapes.items())
    best_per = {mk: min((v for v in t if mk in t[v]), key=lambda v: t[v][mk]) for mk in shapes}
    bptot = sum(t[best_per[mk]][mk] * c for mk, c in shapes.items())
    ranked = sorted(complete, key=tot)
    print(f"\n=== {arch}  N={N}  ({'unweighted' if a.unweighted else 'weighted by calls/step'})")
    print(f"vllm default today : {dtot/1e3:9.3f} ms")
    print(f"best single config : {tot(ranked[0])/1e3:9.3f} ms  {100*(tot(ranked[0])-dtot)/dtot:+.2f}%  {ranked[0]}")
    print(f"best per shape     : {bptot/1e3:9.3f} ms  {100*(bptot-dtot)/dtot:+.2f}%")
    print(f"\n  {'shape':<13}{'calls':>6}{'kernel':>7}  {'vllm default':<14}{'us':>9}  {'best':<14}{'us':>9}{'gain':>7}{'GB/s':>6}"
          f"  {ranked[0]+' us':>18}")
    for mk, c in shapes.items():
        d, b = default[mk], best_per[mk]
        gb = mk[0] * mk[1] * 2 / t[b][mk] / 1e3
        print(f"  {mk[0]}x{mk[1]:<6}{c:>6}{kern[(d, mk)]:>7}  {d:<14}{t[d][mk]:9.1f}  {b:<14}{t[b][mk]:9.1f}"
              f"{100*(t[b][mk]-t[d][mk])/t[d][mk]:+6.1f}%{gb:6.0f}  {complete[ranked[0]][mk]:18.1f}")
    print(f"\n  top {a.top} single configs:")
    for v in ranked[: a.top]:
        print(f"    {v:<14}{tot(v)/1e3:9.3f} ms {100*(tot(v)-dtot)/dtot:+6.2f}%   "
              + " ".join(f"{complete[v][mk]:8.1f}" for mk in shapes))
