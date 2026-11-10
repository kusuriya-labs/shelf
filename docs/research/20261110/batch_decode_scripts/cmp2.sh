#!/bin/bash
# compares dumped outputs of base (split pinned to 1) vs new (auto split) across shapes/rows/bits
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
NEW=$1
for shape in "17408 5120" "5120 17408" "5120 5120" "6144 5120" "1024 5120"; do
 set -- $shape; n=$1; k=$2; shift 2
 for bits in 3 4 5 6; do
  for r in 1 16 17 24 32 33 48 49 64; do
    flock /tmp/r9700.lock /tmp/gemmlab/base --n $n --k $k --rows $r --bits $bits --split 1 --random 1 --iterations 1 --working-mib 1 --dump /tmp/gemmlab/d_base.bin >/dev/null
    out=$(flock /tmp/r9700.lock $NEW --n $n --k $k --rows $r --bits $bits --random 1 --iterations 1 --working-mib 1 --dump /tmp/gemmlab/d_new.bin | grep -o "split=[0-9]*" | head -1)
    python3 -I - "$n k=$k rows=$r bits=$bits $out" <<'P'
import sys,struct
a=open('/tmp/gemmlab/d_base.bin','rb').read(); b=open('/tmp/gemmlab/d_new.bin','rb').read()
import array
x=array.array('H'); x.frombytes(a); y=array.array('H'); y.frombytes(b)
def f(v): return struct.unpack('<f',struct.pack('<I',v<<16))[0]
diff=0; mx=0.0; n=len(x)
rms=(sum(f(p)**2 for p in x)/n)**0.5
for p,q in zip(x,y):
    if p!=q:
        diff+=1; mx=max(mx,abs(f(p)-f(q)))
print('n=%s : elems=%d differing=%d max|d|/rms=%.3e'%(sys.argv[1],n,diff,mx/rms), 'OK' if (len(x)==len(y) and mx/rms<2e-2) else 'CHECK')
P
  done
 done
done
