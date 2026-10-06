#!/usr/bin/env python3
"""Model-agnostic analysis of wvsplitk_sweep CSVs (e.g. from general_sweep.sh).

Every shape counts equally. A config's cost on a shape = time / best time on that shape
(1.00 = optimal). Configs are compared by the geometric mean of that ratio over shapes.

For each N it prints:
  * vllm default heuristic vs best single config vs oracle (best per shape)
  * how well simple rule families recover the oracle: one config per K, per (K, M class),
    per (kernel sml/hf/big, M class)  -> tells you which features a rule list needs
  * winner maps (K rows x M cols): best config per shape, and the default's loss per shape
  * a robust shortlist: configs with the best geomean and the fewest shapes > 3% off

  python3 rule_explorer.py sweep_gfx1100_general_n*.csv [--maps] [--n 2]
"""
import argparse, collections, csv, math

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--maps", action="store_true", help="print K x M winner / loss maps")
ap.add_argument("--n", type=int, nargs="*", help="only these N")
a = ap.parse_args()

def mclass(M):
    return "M<=4K" if M <= 4096 else ("M<=16K" if M <= 16384 else "M>16K")

data = collections.defaultdict(lambda: collections.defaultdict(dict))  # (arch,N) -> shape -> v -> us
default, kern = {}, {}
for f in a.csv:
    for r in csv.DictReader(open(f)):
        key, mk = (r["arch"], int(r["N"])), (int(r["M"]), int(r["K"]))
        if a.n and key[1] not in a.n:
            continue
        if float(r["max_rel_err"]) < 1e-2:
            data[key][mk][r["variant"]] = float(r["us"])
        if r["is_vllm_default"] == "1":
            default[(key, mk)] = r["variant"]
        kern[(key, mk, r["variant"])] = r["kernel"]

def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

def best_group(shapes, ratio, configs):
    """best single config for a group of shapes -> (config, geomean ratio list)"""
    v = min(configs, key=lambda c: gm([ratio[s][c] for s in shapes]))
    return v, [ratio[s][v] for s in shapes]

for key in sorted(data):
    arch, N = key
    sh = data[key]
    shapes = sorted(sh, key=lambda s: (s[1], s[0]))
    configs = set.intersection(*(set(sh[s]) for s in shapes))
    best = {s: min(sh[s].values()) for s in shapes}
    ratio = {s: {c: sh[s][c] / best[s] for c in configs} for s in shapes}
    dflt = [sh[s][default[(key, s)]] / best[s] for s in shapes]
    single, single_r = best_group(shapes, ratio, configs)

    def family(keyf):
        groups = collections.defaultdict(list)
        for s in shapes:
            groups[keyf(s)].append(s)
        out, rs = {}, []
        for g, ss in groups.items():
            v, r = best_group(ss, ratio, configs)
            out[g] = v; rs += r
        return out, rs

    fam = {
        "per K": family(lambda s: s[1]),
        "per (K, M class)": family(lambda s: (s[1], mclass(s[0]))),
        "per (kernel, M class)": family(lambda s: (kern[(key, s, single)], mclass(s[0]))),
    }
    print(f"\n{'=' * 78}\n{arch}  N={N}   {len(shapes)} shapes, {len(configs)} configs"
          f"   (numbers = geomean slowdown vs per-shape best; 1.000 = oracle)")
    print(f"  vllm default today        {gm(dflt):.4f}   worst {max(dflt):.3f}")
    print(f"  best single config        {gm(single_r):.4f}   worst {max(single_r):.3f}   {single}")
    for name, (rules, rs) in fam.items():
        print(f"  best {name:<21}{gm(rs):.4f}   worst {max(rs):.3f}   ({len(set(rules.values()))} distinct configs)")
    print(f"  -> single config vs default: {100 * (gm(single_r) / gm(dflt) - 1):+.2f}% time")

    # robust shortlist
    rows = []
    for c in configs:
        rs = [ratio[s][c] for s in shapes]
        rows.append((gm(rs), sum(r > 1.03 for r in rs), max(rs), c))
    print("  shortlist (geomean, #shapes >3% off, worst):")
    for g, n3, w, c in sorted(rows)[:8]:
        print(f"    {c:<14}{g:.4f}  {n3:3d}  {w:.3f}")

    print("  rules per (kernel, M class):")
    for g, v in sorted(fam["per (kernel, M class)"][0].items()):
        print(f"    {str(g):<22} {v}")

    if a.maps:
        Ms = sorted({s[0] for s in shapes}); Ks = sorted({s[1] for s in shapes})
        print("\n  winner map (rows K, cols M); '.' = shape not run")
        print("  " + " " * 7 + "".join(f"{m:>12}" for m in Ms))
        for k in Ks:
            print(f"  {k:>7}" + "".join(f"{min(sh[(m, k)], key=sh[(m, k)].get):>12}" if (m, k) in sh else f"{'.':>12}" for m in Ms))
        print(f"\n  vllm default: % slower than best (rows K, cols M)")
        for k in Ks:
            print(f"  {k:>7}" + "".join(f"{100 * (sh[(m, k)][default[(key, (m, k))]] / best[(m, k)] - 1):11.1f}%" if (m, k) in sh else f"{'.':>12}" for m in Ms))
        print(f"\n  single config {single}: % slower than best")
        for k in Ks:
            print(f"  {k:>7}" + "".join(f"{100 * (ratio[(m, k)][single] - 1):11.1f}%" if (m, k) in sh else f"{'.':>12}" for m in Ms))
