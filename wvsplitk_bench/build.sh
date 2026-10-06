#!/bin/bash
# Build the standalone harness (run inside the qwen_perf container) and keep the gfx1100 ISA.
#   ./build.sh            -> ./wvsplitk_bench + device ISA in isa/*gfx1100*.s
#   ./build.sh isa d1p1y2u2  -> also print that variant's VGPRs and main-loop memory/sync ops
set -e
cd "$(dirname "$0")"
ARCH=${ARCH:-gfx1100}
HIPCC=${HIPCC:-$(command -v hipcc || echo /opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/bin/hipcc)}
mkdir -p isa
(cd isa && $HIPCC -O3 -std=c++17 --offload-arch=$ARCH --save-temps -o ../wvsplitk_bench ../wvsplitk_bench.hip 2>&1 \
   | grep -v "warning: argument unused" || true)
S=$(ls -t isa/*$ARCH*.s 2>/dev/null | grep -v host | head -1)
# resource summary per kernel: name, vgpr, sgpr, spills
awk '/^[ \t]+\.name:/{name=$2} /^[ \t]+\.vgpr_count:/{v=$2} /^[ \t]+\.sgpr_count:/{s=$2}
     /^[ \t]+\.vgpr_spill_count:/{sp=$2; if (name ~ /wv/) printf "%-4s vgpr=%-4s sgpr=%-4s spill=%s  %s\n", "", v, s, sp, name}' $S \
  | c++filt | sed -E 's/\(int.*//' > isa/resources.txt
echo "built ./wvsplitk_bench; ISA: $S; VGPRs: isa/resources.txt"
if [[ "$1" == "isa" ]]; then
  v=$2  # e.g. d1p1y2u2 -> wv_opt<32, 2, 16, 2, true, true>
  D=${v:1:1}; P=${v:3:1}; Y=${v:5:1}; U=${v:7:1}; b(){ [[ $1 == 1 ]] && echo true || echo false; }
  pat="wv_opt<32, $Y, 16, $U, $(b $D), $(b $P)>"
  grep -F "$pat" isa/resources.txt
  sym=$(grep -oE "_Z6wv_optILi32ELi${Y}ELi16ELi${U}ELb${D}ELb${P}EEv[A-Za-z0-9_]*" $S | head -1)
  awk -v s="$sym:" '$1==s{p=1} p&&/s_endpgm/{print; exit} p' $S \
    | grep -nE "global_load|global_store|s_waitcnt|s_barrier|ds_load|v_dot2|s_cbranch|^\.LBB" \
    > isa/$v.loop.txt
  echo "loop ops -> isa/$v.loop.txt ($(grep -c global_load isa/$v.loop.txt) global_load, $(grep -c v_dot2 isa/$v.loop.txt) v_dot2)"
fi
