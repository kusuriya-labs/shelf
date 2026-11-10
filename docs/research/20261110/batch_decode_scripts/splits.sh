#!/bin/bash
# usage: splits.sh <binary> ; sweeps --split for the mid-row sizes
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
BIN=$1
for shape in "17408 5120" "5120 17408" "5120 5120"; do
  set -- $shape; n=$1; k=$2; shift 2
  for r in 17 24 32 33 40 48 56 64; do
    line="n=$n k=$k rows=$r:"
    for sp in 1 2 3 4 6 8 12; do
      [ $((k/128/sp)) -lt 4 ] && continue
      res=$(flock /tmp/r9700.lock $BIN --n $n --k $k --rows $r --bits 4 --random 1 --working-mib 512 --iterations 16 --split $sp | python3 -I -c "
import sys,re
l=sys.stdin.read()
g=re.search(r'gemm_us=([\d.]+)',l); o=re.search(r'rotate_out_us=([\d.]+)',l)
print('%.0f+%.0f'%(float(g.group(1)),float(o.group(1))))")
      line="$line s$sp=$res"
    done
    echo "$line"
  done
done
