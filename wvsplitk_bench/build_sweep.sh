#!/bin/bash
# Build the config sweep for one batch size N (1..5).
#   ./build_sweep.sh <N> [arch]      arch default: detected via offload-arch / rocminfo, else gfx1100
# Re-extracts vllm's kernels first (set VLLM_SRC to point at another checkout's skinny_gemms.cu).
set -e
cd "$(dirname "$0")"
N=${1:?usage: build_sweep.sh <N> [arch]}
ARCH=${2:-$( (offload-arch 2>/dev/null || rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' ) | head -1)}
ARCH=${ARCH:-gfx1100}
HIPCC=${HIPCC:-$(command -v hipcc || echo /opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/bin/hipcc)}
./extract_kernels.sh ${VLLM_SRC:-}
mkdir -p isa_sweep_n$N
(cd isa_sweep_n$N && $HIPCC -O3 -std=c++17 --offload-arch=$ARCH -DSWEEP_N=$N --save-temps \
   -Rpass-analysis=kernel-resource-usage -o ../wvsplitk_sweep_n$N ../wvsplitk_sweep.hip > build.log 2>&1 \
   || { grep -E "error" build.log | head; exit 1; })
# per-instantiation VGPR / spill / occupancy / LDS summary
awk '/Function Name:/{f=$NF} /VGPRs:/{v=$NF} /ScratchSize/{sc=$NF} /Occupancy/{o=$NF}
     /LDS Size/{ if (f ~ /wvSplitK/) print f, "vgpr=" v, "scratch=" sc, "occ=" o, "lds=" $NF }' \
  isa_sweep_n$N/build.log | c++filt | sed -E 's/\(int, int.*\)//' > isa_sweep_n$N/resources.txt
echo "built ./wvsplitk_sweep_n$N for $ARCH; resources: isa_sweep_n$N/resources.txt"
