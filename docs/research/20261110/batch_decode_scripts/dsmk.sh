#!/bin/bash
# usage: dsmk.sh <binary> <tag> <N> <K> [extra args]   (K=0 -> no speculation)
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
export SHELF_BENCH_BINARY=$1 SHELF_DFLASH2_SERVE_STATS=1
D=/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2
TAG=$2; N=$3; K=$4; shift 4
SPEC=""; [ "$K" != 0 ] && SPEC="--dflash2-model-dir $D --dflash2-drafts $K"
cd /home/homelab1
ERRF=/tmp/agentx/k_$TAG.err GENF=/tmp/agentx/k_$TAG.gen flock /tmp/r9700.lock timeout 1500 python3 -I /tmp/agentx/conc3.py $N 160 --kv-cache-dtype fp8 --max-num-batched-tokens 2048 $SPEC "$@" 2>&1 | tail -1 | sed 's/"extra".*"total_tokens"/"total_tokens"/' | cut -c1-130 | sed "s/^/$TAG /" | tee -a /tmp/agentx/k_log
