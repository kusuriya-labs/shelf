#!/bin/bash
# usage: sweep.sh <binary> <tag> [extra args]; prints gemm_us per (shape, rows)
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
BIN=$1; TAG=$2; shift 2
for shape in "17408 5120" "5120 17408" "5120 5120" "10240 5120"; do
  set -- $shape; n=$1; k=$2; shift 2
  for r in 1 8 16 17 24 32 33 48 64; do
    flock /tmp/r9700.lock $BIN --n $n --k $k --rows $r --bits 4 --random 1 --working-mib 512 --iterations 16 $EXTRA | python3 -I -c "
import sys,re
l=sys.stdin.read()
m=re.search(r'split=(\d+)',l); g=re.search(r'gemm_us=([\d.]+)',l); t=re.search(r'total_us=([\d.]+)',l)
print('$TAG n=$n k=$k rows=$r split=%s gemm_us=%s total_us=%s'%(m.group(1),g.group(1),t.group(1)))"
  done
done
