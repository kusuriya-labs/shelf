#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 OPENBLAS_NUM_THREADS=1
unset SHELF_QUALITY_VERIFY SHELF_QUALITY_BATCH8 SHELF_GDN_VERIFY_CHUNKED SHELF_GDN_VERIFY_CHUNKED_EXACT SHELF_GDN_RECURRENCE_EXACT
model=/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24
root=/tmp/gdn-quality-full
out=/tmp/gdn-quality-long
mkdir -p "$out/corpus" "$out/dumps"
/tmp/shelfvenv/bin/python - "$root/corpus/reasoning.psktok" "$out/corpus/reasoning.psktok" <<'PY'
import hashlib, struct, sys
from pathlib import Path
raw = Path(sys.argv[1]).read_bytes()
magic, version, count = struct.unpack('<8sIQ', raw[:20])
assert magic == b'PSKLDTOK' and version == 1
ids = struct.unpack(f'<{count}i', raw[20:])
ids = (ids * ((4233 + count - 1) // count))[:4233]
payload = struct.pack('<8sIQ', magic, version, len(ids)) + struct.pack(f'<{len(ids)}i', *ids)
Path(sys.argv[2]).write_bytes(payload)
print('extended corpus SHA256', hashlib.sha256(payload).hexdigest(), flush=True)
PY
for corpus in prose code reasoning japanese; do
  echo "RUN $corpus chunk8 $(date -Is)"
  flock /tmp/r9700.lock env SHELF_QUALITY_BATCH8=1 SHELF_QUALITY_VERIFY=1 SHELF_GDN_VERIFY_CHUNKED=1 /tmp/gdn-quality-bench quality --model-dir "$model" --tokens-file "$root/corpus/$corpus.psktok" --context 128 --positions 2048 --device 0 --dump "$root/dumps/$corpus-chunk8.psq"
done
for context in 128 132; do
  for mode in base serial; do
    envs=()
    [[ $mode == serial ]] && envs=(SHELF_QUALITY_VERIFY=1)
    echo "RUN reasoning long context=$context $mode $(date -Is)"
    flock /tmp/r9700.lock env "${envs[@]}" /tmp/gdn-quality-long-bench quality --model-dir "$model" --tokens-file "$out/corpus/reasoning.psktok" --context "$context" --positions 4096 --device 0 --dump "$out/dumps/reasoning-${context}-${mode}.psq"
  done
done
echo "ALL DONE $(date -Is)"
