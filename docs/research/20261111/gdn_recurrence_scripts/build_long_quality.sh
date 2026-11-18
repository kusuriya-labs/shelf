#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source=src/apps/bench/quality.hip
patch=docs/research/20261111/gdn_recurrence_scripts/quality_verify.patch
if ! git diff --quiet -- "$source"; then echo 'quality source already modified' >&2; exit 1; fi
applied=0
cleanup() { if (( applied )); then git restore -- "$source"; fi; }
trap cleanup EXIT
git apply --check "$patch"
git apply "$patch"
applied=1
python3 -c "from pathlib import Path; p=Path('$source'); s=p.read_text(); assert 'positions, 2048u' in s; p.write_text(s.replace('positions, 2048u', 'positions, 8192u').replace(' || context % 8u != 0u', ''))"
nice -n 10 ninja -C build-gfx1201 -j32 shelf-bench
cp build-gfx1201/shelf-bench /tmp/gdn-quality-long-bench
git restore -- "$source"
applied=0
nice -n 10 ninja -C build-gfx1201 -j32 shelf-bench
