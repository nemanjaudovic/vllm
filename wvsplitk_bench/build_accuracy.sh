#!/bin/bash
# Build the fp32-dot vs native-dot accuracy harness.   ./build_accuracy.sh [arch]
set -e
cd "$(dirname "$0")"
ARCH=${1:-$( (offload-arch 2>/dev/null || rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+') | head -1)}
ARCH=${ARCH:-gfx1100}
HIPCC=${HIPCC:-$(command -v hipcc || echo /opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/bin/hipcc)}
[[ -n "$SKIP_EXTRACT" ]] || ./extract_kernels.sh ${VLLM_SRC:-}
$HIPCC -O3 -std=c++17 --offload-arch=$ARCH -o wvsplitk_accuracy wvsplitk_accuracy.hip
echo "built ./wvsplitk_accuracy for $ARCH"
