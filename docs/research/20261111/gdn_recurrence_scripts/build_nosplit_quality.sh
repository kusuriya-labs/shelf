#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
quality=src/apps/bench/quality.hip
dispatch=src/shelf/inference/execution/paged_attention_dispatch.hip
patch=docs/research/20261111/gdn_recurrence_scripts/quality_verify.patch
if ! git diff --quiet -- "$quality" "$dispatch"; then echo 'source already modified' >&2; exit 1; fi
cleanup() { git restore -- "$quality" "$dispatch"; }
trap cleanup EXIT
git apply --check "$patch"
git apply "$patch"
python3 -c "from pathlib import Path; p=Path('$dispatch'); s=p.read_text(); old='ctx.spec.max_visible_tokens >= kPagedAttentionSplitMinVisible'; assert s.count(old)==1; p.write_text(s.replace(old, 'ctx.spec.max_visible_tokens >= 65536u'))"
nice -n 10 ninja -C build-gfx1201 -j32 shelf-bench
cp build-gfx1201/shelf-bench /tmp/gdn-quality-nosplit-bench
cleanup
trap - EXIT
nice -n 10 ninja -C build-gfx1201 -j32 shelf-bench
