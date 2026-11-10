#!/bin/bash
export PATH=/tmp/shelfvenv/bin:$PATH LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
W=/home/homelab1/coding-local/shelf/.worktrees/integ-gemm-dflash/build-gfx1201
BIN=$W/shelf-compute
cd /home/homelab1/coding-local
run() { tag=$1; shift; flock /tmp/r9700.lock timeout 1700 python3 -I /tmp/agentx/val.py --bin $BIN --tag $tag --err /tmp/agentx/val_$tag.err "$@" 2>&1 | tail -1 >> /tmp/agentx/val_log; sleep 6; }
srv() { tag=$1; shift; flock /tmp/r9700.lock timeout 1700 python3 -I /tmp/agentx/srv.py --server $W/shelf-server --err /tmp/agentx/val_$tag.err "$@" 2>&1 | tail -1 | sed "s/^/SERVER $tag /" >> /tmp/agentx/val_log; sleep 8; }
# harness sanity (short context, known: spec N=8 ~233, nospec N=16 ~286)
run d0_spec8   --mode steady --n 8  --spec 7
run d0_nospec16 --mode steady --n 16
# (1) prefill arrival while 8 requests decode
run a1_mix2048 --mode mix --n 8 --new 500 --inject 8192 --mnbt 2048
run a2_mix512  --mode mix --n 8 --new 500 --inject 8192 --mnbt 512
# (2) long-context batch decode (shared prefix, prefix cache)
run b1_l16k_n8_nospec  --mode steady --n 8  --prefix 16384 --pc 1
run b2_l16k_n8_spec    --mode steady --n 8  --prefix 16384 --pc 1 --spec 7
run b3_l16k_n16_nospec --mode steady --n 16 --prefix 16384 --pc 1
run b4_l64k_n4_nospec  --mode steady --n 4  --prefix 65536 --pc 1
run b5_l64k_n4_spec    --mode steady --n 4  --prefix 65536 --pc 1 --spec 7
# temperature=1 and c4v4
run e1_t1_spec8        --mode steady --n 8  --spec 7 --temp 1
run e2_c4v4_spec8      --mode steady --n 8  --spec 7 --kv octave:c4v4
run e3_c4v4_t1_spec8   --mode steady --n 8  --spec 7 --kv octave:c4v4 --temp 1
run e4_c4v4_t1_nospec16 --mode steady --n 16 --kv octave:c4v4 --temp 1
run e5_c4v4_t1_l16k_nospec --mode steady --n 8 --prefix 16384 --pc 1 --kv octave:c4v4 --temp 1
run e6_c4v4_t1_l16k_spec   --mode steady --n 8 --prefix 16384 --pc 1 --kv octave:c4v4 --temp 1 --spec 7
# (3) through shelf-server
srv s1_srv_nospec16 --n 16
srv s2_srv_spec8    --n 8 --spec 7
echo DONE >> /tmp/agentx/val_log
