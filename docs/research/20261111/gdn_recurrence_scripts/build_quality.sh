#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
patch=docs/research/20261111/gdn_recurrence_scripts/quality_verify.patch
source=src/apps/bench/quality.hip
if ! git diff --quiet -- "$source"; then echo 'quality source already modified' >&2; exit 1; fi
applied=0
cleanup() { if (( applied )); then git apply -R "$patch"; fi; }
trap cleanup EXIT
git apply --check "$patch"
git apply "$patch"
applied=1
nice -n 10 ninja -C build-gfx1201 -j32 shelf-bench
cp build-gfx1201/shelf-bench /tmp/gdn-quality-bench
git apply -R "$patch"
applied=0
nice -n 10 ninja -C build-gfx1201 -j32 shelf-bench
