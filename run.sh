#!/bin/bash
./build/shelf-server \
    --model-dir models/Qwen3.8-Flash-Next-exl3 \
    --dflash2-model-dir models/Qwen3.8-Flash-Next-DFlash2 --dflash2-drafts 6 \
    --device 2,3 \
    --tensor-parallel 2 \
    --max-concurrent-requests 2 \
    --max-seq-len 262144 \
    --kv-cache-dtype fp8 \
    --port 8123

    # --max-seq-len 524288 --rope-factor 2 \
