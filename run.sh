#!/bin/bash
./build/shelf-launcher server \
    --model-dir models/Qwen3.8-27B-FP8 \
    --eagle4-model-dir models/Qwen3.8-27B-EAGLE4 --eagle4-drafts 7 --eagle4-lm-head draft \
    --device 0,1 \
    --tp 2 \
    --max-concurrent-requests 2 \
    --max-seq-len 262144 \
    --kv-cache-dtype fp8 \
    --port 8123
