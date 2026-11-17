#!/bin/bash
export SHELF_EXL3_MUL24_PREFILL_AB=B
# export SHELF_EXL3_MUL24_AUTO=1
# ./build/shelf-bench pp \
#     --model-dir models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw \
#     --device 0,1 --tensor-parallel 2 --warmup 1 \
#     --prompt-tokens 2048 --kv-cache-dtype fp8
./build/shelf-bench pp \
    --model-dir models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw \
    --device 0 --warmup 1 \
    --prompt-tokens 65536 --kv-cache-dtype fp8
# ./build/shelf-bench tg \
#     --model-dir models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw \
#     --device 0 --warmup 1 \
#     --context 3072 --tokens 256 --kv-cache-dtype fp8
