#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
unset SHELF_QUALITY_VERIFY SHELF_QUALITY_BATCH8 SHELF_GDN_VERIFY_CHUNKED SHELF_GDN_VERIFY_CHUNKED_EXACT SHELF_GDN_RECURRENCE_EXACT
model=/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24
out=/tmp/gdn-quality-long
for context in 2040 4080; do
  echo "RUN prefill context=$context $(date -Is)"
  flock /tmp/r9700.lock /tmp/gdn-quality-prefill-bench quality --model-dir "$model" --tokens-file "$out/corpus/reasoning.psktok" --context "$context" --positions 32 --device 0 --dump "$out/dumps/reasoning-prefill-$context.psq"
done
echo "ALL DONE $(date -Is)"
