#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
export OPENBLAS_NUM_THREADS=1
export SHELF_MODEL_DIR=/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24
export SHELF_DFLASH2_MODEL_DIR=/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2
exec /tmp/shelfvenv/bin/python docs/research/20261111/draft_forest_scripts/m11_measure.py
