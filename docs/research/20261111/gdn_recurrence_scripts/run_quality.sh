#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
export OPENBLAS_NUM_THREADS=1
unset SHELF_QUALITY_VERIFY SHELF_QUALITY_BATCH8 SHELF_GDN_VERIFY_CHUNKED SHELF_GDN_VERIFY_CHUNKED_EXACT SHELF_GDN_RECURRENCE_EXACT
model=/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24
root=${GDN_QUALITY_DIR:-/tmp/gdn-quality-full}
mkdir -p "$root/corpus" "$root/dumps" "$root/reports"
/tmp/shelfvenv/bin/python docs/research/20261111/gdn_recurrence_scripts/quality_corpus.py "$root/corpus" | tee "$root/corpus.txt"
for corpus in prose code reasoning japanese; do
  for mode in base batch8 serial chunk exact serial8 exact8; do
    case "$mode" in
      base) envs=();;
      batch8) envs=(SHELF_QUALITY_BATCH8=1);;
      serial) envs=(SHELF_QUALITY_VERIFY=1);;
      chunk) envs=(SHELF_QUALITY_VERIFY=1 SHELF_GDN_VERIFY_CHUNKED=1);;
      exact) envs=(SHELF_QUALITY_VERIFY=1 SHELF_GDN_VERIFY_CHUNKED=1 SHELF_GDN_VERIFY_CHUNKED_EXACT=1);;
      serial8) envs=(SHELF_QUALITY_BATCH8=1 SHELF_QUALITY_VERIFY=1);;
      exact8) envs=(SHELF_QUALITY_BATCH8=1 SHELF_QUALITY_VERIFY=1 SHELF_GDN_VERIFY_CHUNKED=1 SHELF_GDN_VERIFY_CHUNKED_EXACT=1);;
    esac
    dump="$root/dumps/$corpus-$mode.psq"
    echo "RUN $corpus $mode $(date -Is)"
    flock /tmp/r9700.lock env "${envs[@]}" /tmp/gdn-quality-bench quality --model-dir "$model" --tokens-file "$root/corpus/$corpus.psktok" --context 128 --positions 2048 --device 0 --dump "$dump"
    if [[ $mode != base ]]; then
      /tmp/shelfvenv/bin/python docs/research/20261111/gdn_recurrence_scripts/quality_score.py "$root/dumps/$corpus-base.psq" "$dump" > "$root/reports/$corpus-$mode.json"
    fi
  done
done
