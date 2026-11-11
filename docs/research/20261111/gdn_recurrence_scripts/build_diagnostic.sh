#!/bin/bash
# Execute only after the ordinary measurements have finished.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
patch=docs/research/20261111/gdn_recurrence_scripts/verify_logits_diagnostic.patch
source=src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp
if ! git diff --quiet -- "$source"; then echo 'source already modified' >&2; exit 1; fi
applied=0
cleanup() { if (( applied )); then git apply -R "$patch"; fi; }
trap cleanup EXIT
git apply --check "$patch"
git apply "$patch"
applied=1
nice -n 10 ninja -C build-gfx1201 -j32 shelf-compute
cp build-gfx1201/shelf-compute /tmp/gdn-divergence-compute-diagnostic
git apply -R "$patch"
applied=0
nice -n 10 ninja -C build-gfx1201 -j32 shelf-compute
