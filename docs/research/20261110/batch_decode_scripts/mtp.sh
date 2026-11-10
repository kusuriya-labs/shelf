#!/bin/bash
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
export SHELF_BENCH_BINARY=/home/homelab1/coding-local/shelf/.worktrees/dflash2-session-mem/build-gfx1201/shelf-compute
cd /home/homelab1
for N in 1 4 8 16 32; do for K in 4; do
 ERRF=/tmp/agentx/mtp_N${N}_K$K.err GENF=/tmp/agentx/mtp_N${N}_K$K.gen flock /tmp/r9700.lock timeout 1500 python3 -I /tmp/agentx/conc3.py $N 160 --kv-cache-dtype fp8 --max-num-batched-tokens 2048 --mtp --mtp-drafts $K 2>&1 | tail -1 | sed 's/"extra".*"total_tokens"/"total_tokens"/' | cut -c1-200 | sed "s/^/MTP N=$N K=$K /" >> /tmp/agentx/mtp_log
done; done; echo DONE >> /tmp/agentx/mtp_log
