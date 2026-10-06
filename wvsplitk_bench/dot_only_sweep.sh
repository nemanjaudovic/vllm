#!/bin/bash
# Native-bf16-dot-only check: for every shape in general_shapes.txt run vllm's CURRENT tile config
# (gfx1151 rules on gfx1151, generic gfx1x heuristic elsewhere) with fp32 dot vs native dot.
#   ./dot_only_sweep.sh [arch] [Ns="1 2 3 4 5"] [rounds=5]
# Output: sweep_<arch>_dotonly_n<N>.csv, summary via dot_compare.py
set -e
cd "$(dirname "$0")"
ARCH=${1:-$( (offload-arch 2>/dev/null || rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+') | head -1)}
NS=${2:-1 2 3 4 5}; ROUNDS=${3:-5}
LOG=dotonly_sweep_${ARCH}.log
./extract_kernels.sh ${VLLM_SRC:-} | tee $LOG
for N in $NS; do SKIP_EXTRACT=1 ./build_sweep.sh $N $ARCH >> $LOG 2>&1 & done; wait
for N in $NS; do
  [[ -x wvsplitk_sweep_n$N ]] || { echo "build for N=$N failed, see isa_sweep_n$N/build.log"; exit 1; }
  (amd-smi metric -u 2>/dev/null | grep -m1 GFX_ACTIVITY || true) | tee -a $LOG
  ./wvsplitk_sweep_n$N --dot-only --shapes "$(cat general_shapes.txt)" --rounds $ROUNDS \
      --csv sweep_${ARCH}_dotonly_n$N.csv >> $LOG 2>&1
done
grep -c "WRONG RESULT" $LOG && echo "!! wrong results, see $LOG" || true
python3 dot_compare.py $(for N in $NS; do echo sweep_${ARCH}_dotonly_n$N.csv; done)
