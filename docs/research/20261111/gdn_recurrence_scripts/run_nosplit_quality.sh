#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
unset SHELF_QUALITY_VERIFY SHELF_QUALITY_BATCH8 SHELF_GDN_VERIFY_CHUNKED SHELF_GDN_VERIFY_CHUNKED_EXACT SHELF_GDN_RECURRENCE_EXACT
model=/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24
root=/tmp/gdn-quality-full
out=/tmp/gdn-quality-nosplit
mkdir -p "$out"
for mode in base serial; do
  envs=()
  [[ $mode == serial ]] && envs=(SHELF_QUALITY_VERIFY=1)
  echo "RUN reasoning $mode $(date -Is)"
  flock /tmp/r9700.lock env "${envs[@]}" /tmp/gdn-quality-nosplit-bench quality --model-dir "$model" --tokens-file "$root/corpus/reasoning.psktok" --context 128 --positions 2048 --device 0 --dump "$out/reasoning-$mode.psq"
done
echo "ALL DONE $(date -Is)"
