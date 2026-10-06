#!/bin/bash
# Pull vllm's wvSplitK device kernels (sml / hf / big + min__ + mindiv) out of
# csrc/rocm/skinny_gemms.cu into vllm_kernels.inc, WITHOUT the DOT2C macro
# (the sweep harness defines DOT2C itself: fp32 path and native-dot path).
#   ./extract_kernels.sh [path/to/skinny_gemms.cu]
set -e
cd "$(dirname "$0")"
# default: this dir inside a vllm checkout (../csrc), else a sibling vllm checkout (../vllm/csrc)
SRC=${1:-$(ls ../csrc/rocm/skinny_gemms.cu ../vllm/csrc/rocm/skinny_gemms.cu 2>/dev/null | head -1)}
# start: first line after the DOT2C #if/#elif/#endif block; end: line before host wvSplitK()
a=$(grep -n '^#if defined(__HIP__GFX9__) && !defined(__HIP__GFX1X__)' "$SRC" | head -1 | cut -d: -f1)
e=$(awk -v a="$a" 'NR>a && /^#endif/{print NR; exit}' "$SRC")
b=$(grep -n '^torch::Tensor wvSplitK(' "$SRC" | cut -d: -f1)
[[ -n "$a" && -n "$e" && -n "$b" ]] || { echo "markers not found in $SRC"; exit 1; }
{ echo "// AUTO-GENERATED from $SRC lines $((e+1))-$((b-1)) by extract_kernels.sh; do not edit"
  sed -n "$((e+1)),$((b-1))p" "$SRC"; } > vllm_kernels.inc
echo "vllm_kernels.inc <- $SRC lines $((e+1))-$((b-1)) ($(grep -c '__global__' vllm_kernels.inc) __global__ decls)"
