#!/bin/bash
# General (model-agnostic) sweep: all N on the M x K grid in general_shapes.txt.
#   ./general_sweep.sh [arch] [Ns="1 2 3 4 5"] [rounds=3] [target_ms=10]
#   e.g. ./general_sweep.sh gfx1201 1      (N=1 only)
# Analyze: python3 rule_explorer.py sweep_<arch>_general_n*.csv --maps
# Output: sweep_<arch>_general_n<N>.csv + general_sweep_<arch>.log
set -e
cd "$(dirname "$0")"
ARCH=${1:-$( (offload-arch 2>/dev/null || rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+') | head -1)}
NS=${2:-1 2 3 4 5}; ROUNDS=${3:-3}; TMS=${4:-10}
LOG=general_sweep_${ARCH}.log
SHAPES=$(cat general_shapes.txt)
echo "arch=$ARCH Ns=$NS rounds=$ROUNDS target_ms=$TMS shapes=$(echo $SHAPES | tr , '\n' | wc -l) $(date)" | tee $LOG
./extract_kernels.sh ${VLLM_SRC:-} | tee -a $LOG   # once; parallel builds below skip it
for N in $NS; do SKIP_EXTRACT=1 ./build_sweep.sh $N $ARCH >> $LOG 2>&1 & done; wait
for N in $NS; do [[ -x wvsplitk_sweep_n$N ]] || { echo "build for N=$N failed, see isa_sweep_n$N/build.log"; exit 1; }; done
for N in $NS; do
  echo "--- N=$N $(date)" | tee -a $LOG
  (amd-smi metric -u 2>/dev/null | grep -m1 GFX_ACTIVITY || true) | tee -a $LOG   # should be ~0%
  ./wvsplitk_sweep_n$N --shapes "$SHAPES" --rounds $ROUNDS --target-ms $TMS \
      --csv sweep_${ARCH}_general_n$N.csv >> $LOG 2>&1
done
echo "done $(date)" | tee -a $LOG
