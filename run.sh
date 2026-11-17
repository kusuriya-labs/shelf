#!/bin/bash
./build/shelf-server \
    --model-dir models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4 \
    --prefix-cache-strategy radix \
    --dflash2-model-dir models/Qwen3.8-27B-DFlash2-FP8 --dflash2-drafts 6 --dflash2-tree-budget 0 \
    --max-concurrent-requests 1 \
    --tensor-parallel 2 --device 2,3 \
    --max-seq-len 262144 \
    --kv-cache-dtype octave:c5v5 \
    --port 8123

# ./build/shelf-server \
#     --model-dir models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw \
#     --device 2,3 \
#     --tensor-parallel 2 \
#     --prefix-cache-strategy radix \
#     --max-concurrent-requests 2 \
#     --max-seq-len 262144 \
#     --kv-cache-dtype octave:c5v5 \
#     --port 8123
