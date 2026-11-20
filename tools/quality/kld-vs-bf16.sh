#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf '%s\n' '使い方: kld-vs-bf16.sh [--already-locked] 構成名 [Python オプション]' 'R9700 の /tmp/r9700.lock を取得します。with-gpu の内側では --already-locked を指定してください。' 'R9700 の B 構成は約196秒（実測1回）。構成と環境で変動します。実測値は docs/perf/quality-bf16/results.md を参照。'
    exec "${PYTHON:-/tmp/shelfvenv/bin/python}" "$root/tools/quality/kld_vs_bf16.py" --help
fi
if [[ ${1:-} == --already-locked ]]; then
    shift
else
    exec flock /tmp/r9700.lock "$0" --already-locked "$@"
fi
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/core-10.2/lib:/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib}"
exec "${PYTHON:-/tmp/shelfvenv/bin/python}" "$root/tools/quality/kld_vs_bf16.py" "$@"
