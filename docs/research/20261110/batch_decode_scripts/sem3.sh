#!/bin/bash
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
export SHELF_BENCH_BINARY=/home/homelab1/coding-local/shelf/.worktrees/integ-gemm-dflash/build-gfx1201/shelf-compute SHELF_QWEN35_PERF_TIMING=1 SHELF_BATCH_STEPWALL=1 SHELF_DFLASH2_SERVE_STATS=1
D=/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2
cd /home/homelab1
for cfg in "16 16" "32 32"; do set -- $cfg; N=$1; S=$2
 SHELF_BATCH_SEM_DUMP=/tmp/agentx/sem3/n$N ERRF=/tmp/agentx/sem3/n$N.err flock /tmp/r9700.lock timeout 900 python3 -I /tmp/agentx/conc3.py $N 160 --kv-cache-dtype fp8 --max-num-batched-tokens 2048 --dflash2-model-dir $D --dflash2-drafts 2 --dflash2-max-sessions $S >/dev/null 2>&1
done; touch /tmp/agentx/sem3/done
