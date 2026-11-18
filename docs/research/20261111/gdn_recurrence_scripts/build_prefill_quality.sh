#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source=src/apps/bench/quality.hip
patch=docs/research/20261111/gdn_recurrence_scripts/quality_verify.patch
if ! git diff --quiet -- "$source"; then echo 'source already modified' >&2; exit 1; fi
trap 'git restore -- "$source"' EXIT
git apply --check "$patch"
git apply "$patch"
python3 - "$source" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]); s = p.read_text()
replacements = {
'const bool verify = std::getenv("SHELF_QUALITY_VERIFY") != nullptr;': 'const bool verify = false;',
'const uint32_t rows = verify ? 8u : 1u;': 'const uint32_t rows = 8u;',
'(verify ? 8u : 1u)': '8u',
'verify ? shelf::runtime::ExecutionClass::SPEC_VERIFY : shelf::runtime::ExecutionClass::DECODE': 'shelf::runtime::ExecutionClass::PREFILL',
'batch.num_decode_requests = verify ? 0u : concurrency;': 'batch.num_decode_requests = 0u;\n        batch.num_prefill_requests = concurrency;',
}
for old, new in replacements.items():
    assert s.count(old) == 1, old
    s = s.replace(old, new)
p.write_text(s)
PY
nice -n 10 ninja -C build-gfx1201 -j32 shelf-bench
cp build-gfx1201/shelf-bench /tmp/gdn-quality-prefill-bench
git restore -- "$source"
trap - EXIT
nice -n 10 ninja -C build-gfx1201 -j32 shelf-bench
