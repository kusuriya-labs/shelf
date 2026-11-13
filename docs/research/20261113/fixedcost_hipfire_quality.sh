#!/bin/bash
set -euo pipefail
root=$(pwd)
model=/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for direction in ja-en en-ja; do
  flock /tmp/r9700.lock timeout -k 5s 270s /tmp/venv-plamo/bin/python \
    "$root/docs/research/20261112/plamo_quality.py" \
    --reference /tmp/plamo-expanded-reference.pt --model "$model" \
    --bench "$root/build-gfx1201/shelf-bench" --workdir "/tmp/fc_hipfire/quality-$direction" \
    --batch-size 1 --prefix "heldout-$direction-" \
    > "/tmp/fc_hipfire/quality-$direction.log" 2>&1
done
