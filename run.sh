#!/bin/bash
./build/shelf-launcher server \
    --model-dir models/Qwen3.8-27B-FP8 \
    --dflash2-model-dir models/Qwen3.8-27B-DFlash2 --dflash2-drafts 7 \
    --device 0,1 \
    --tp 2 \
    --max-concurrent-requests 2 \
    --max-seq-len 262144 \
    --kv-cache-dtype fp8 \
    --port 8123
