#!/bin/bash
# usage: gmr.sh <binary> <tag> <Ns...> -- [extra args]
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
export SHELF_BENCH_BINARY=$1; TAG=$2; shift 2
NS=(); while [ "$1" != "--" ]; do NS+=($1); shift; done; shift
cd /home/homelab1
for N in "${NS[@]}"; do
 ERRF=/tmp/agentx/gmr_${TAG}_N$N.err GENF=/tmp/agentx/gmr_${TAG}_N$N.gen flock /tmp/r9700.lock timeout 1500 python3 -I /tmp/agentx/conc3.py $N 160 --kv-cache-dtype fp8 --max-num-batched-tokens 2048 "$@" 2>&1 | tail -1 | sed 's/"extra".*"total_tokens"/"total_tokens"/' | cut -c1-120 | sed "s/^/$TAG N=$N /" >> /tmp/agentx/gmr_log
done
