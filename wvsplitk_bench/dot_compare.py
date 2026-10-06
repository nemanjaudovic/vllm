#!/usr/bin/env python3
"""Dot-only comparison: vllm's CURRENT tile choice per shape, fp32 bf16 dot vs native v_dot2_f32_bf16.
(Everything but DOT2C identical; this is exactly what a 'native bf16 dot only' change does.)

  python3 dot_compare.py results/<arch>/general_n*.csv            # every shape weighted equally
  python3 dot_compare.py results/<arch>/qwen_n*.csv --weighted     # weight by calls/step
  [--per-shape]  list every shape
"""
import argparse, collections, csv, math

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--weighted", action="store_true")
ap.add_argument("--per-shape", action="store_true")
a = ap.parse_args()
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))

print(f"{'arch':<8}{'N':>2}{'shapes':>7}  {'native/fp32 time':>16}{'min':>8}{'max':>8}{'#slower':>8}{'#>1% slower':>12}"
      f"  {'max rel err fp32':>16}{'native':>9}")
for f in a.csv:
    rows = list(csv.DictReader(open(f)))
    t = {(r["M"], r["K"], r["variant"]): r for r in rows}
    out = []
    for r in rows:
        if r["is_vllm_default"] != "1":
            continue
        nd = t.get((r["M"], r["K"], "nd_" + r["variant"][3:]))
        if nd is None:
            continue
        out.append((float(nd["us"]) / float(r["us"]), r, nd))
    if not out:
        continue
    ratios = [x[0] for x in out]
    if a.weighted:
        w = [int(x[1]["calls"]) for x in out]
        agg = sum(float(x[2]["us"]) * c for x, c in zip(out, w)) / sum(float(x[1]["us"]) * c for x, c in zip(out, w))
    else:
        agg = gm(ratios)
    r0 = out[0][1]
    print(f"{r0['arch']:<8}{r0['N']:>2}{len(out):>7}  {100*(agg-1):+15.2f}%{100*(min(ratios)-1):+7.1f}%{100*(max(ratios)-1):+7.1f}%"
          f"{sum(x > 1 for x in ratios):>8}{sum(x > 1.01 for x in ratios):>12}"
          f"  {max(float(x[1]['max_rel_err']) for x in out):16.2e}{max(float(x[2]['max_rel_err']) for x in out):9.2e}")
    if a.per_shape:
        for ratio, fp, nd in sorted(out, key=lambda x: x[0]):
            print(f"     {fp['M']:>7}x{fp['K']:<6} {fp['variant']:<12}{float(fp['us']):9.1f} -> {float(nd['us']):9.1f} us  {100*(ratio-1):+5.1f}%")
