#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
export SHELF_MODEL_DIR=/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24 SHELF_DFLASH2_MODEL_DIR=/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2 SHELF_BENCH_BINARY=$PWD/build-gfx1201/shelf-compute
export SHELF_GDN_VERIFY_CHUNKED=1 SHELF_GDN_VERIFY_CHUNKED_EXACT=1
unset SHELF_GDN_RECURRENCE_EXACT SHELF_GDN_RECURRENCE_LOSSY
for n in 4 8 16 32; do
 for rep in 1 2; do
  echo "START batch $n $rep"
  export ERRF=/tmp/gdn-exact-batch-$n-$rep.err GENF=/tmp/gdn-exact-batch-$n-$rep.json
  flock /tmp/r9700.lock /tmp/venv-so/bin/python -I docs/research/20261111/gdn_recurrence_scripts/batch_benchmark.py "$n" 160 --kv-cache-dtype fp8 --max-num-batched-tokens 2048 --dflash2-model-dir "$SHELF_DFLASH2_MODEL_DIR" --dflash2-drafts 7
 done
done
for i in 0 1 2; do
 for rep in 0 1 2; do
  flock /tmp/r9700.lock /tmp/venv-so/bin/python -I docs/research/20261111/gdn_recurrence_scripts/exact_round.py exact "$i" "$rep"
 done
done
