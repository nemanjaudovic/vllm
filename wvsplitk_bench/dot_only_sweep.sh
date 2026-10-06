#!/bin/bash
# Native-bf16-dot-only check: for every shape in general_shapes.txt run vllm's CURRENT tile config
# (gfx1151 rules on gfx1151, generic gfx1x heuristic elsewhere) with fp32 dot vs native dot.
#   ./dot_only_sweep.sh [arch] [Ns="1 2 3 4 5"] [rounds=5]
#   SWEEP_ARGS="--pad" TAG=_pad ./dot_only_sweep.sh ...   (with PR #55090 row-stride padding)
# Output: sweep_<arch>_dotonly_n<N>.csv, summary via dot_compare.py
set -e
cd "$(dirname "$0")"
ARCH=${1:-$( (offload-arch 2>/dev/null || rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+') | head -1)}
NS=${2:-1 2 3 4 5}; ROUNDS=${3:-5}
LOG=dotonly_sweep_${ARCH}.log
./extract_kernels.sh ${VLLM_SRC:-} | tee $LOG
echo "building N=$NS in parallel (reduced dot-only build; GPU stays idle meanwhile)..." | tee -a $LOG
for N in $NS; do
  ( t0=$SECONDS; SKIP_EXTRACT=1 SWEEP_FLAGS=-DDOT_ONLY_BUILD ./build_sweep.sh $N $ARCH >> $LOG 2>&1 \
      && echo "  built N=$N in $((SECONDS - t0)) s" || echo "  build N=$N FAILED, see isa_sweep_n$N/build.log" ) | tee -a $LOG &
done; wait
for N in $NS; do
  [[ -x wvsplitk_sweep_n$N ]] || { echo "build for N=$N failed, see isa_sweep_n$N/build.log"; exit 1; }
  (amd-smi metric -u 2>/dev/null | grep -m1 GFX_ACTIVITY || true) | tee -a $LOG
  ./wvsplitk_sweep_n$N --dot-only --shapes "$(cat general_shapes.txt)" --rounds $ROUNDS \
      $SWEEP_ARGS --csv sweep_${ARCH}_dotonly${TAG:-}_n$N.csv 2>&1 | tee -a $LOG
done
grep -c "WRONG RESULT" $LOG && echo "!! wrong results, see $LOG" || true
python3 dot_compare.py $(for N in $NS; do echo sweep_${ARCH}_dotonly${TAG:-}_n$N.csv; done)
