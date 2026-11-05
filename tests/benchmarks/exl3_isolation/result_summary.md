# EXL3 mul1 isolation benchmark — first 3 stages

M=2048, K=N=5120, groups=1, split-K=1, gfx1201, 10 warmup + 50 samples.

| Stage | kernel ms | tok/s (single linear) | TFLOPS | vs rocBLAS |
| --- | ---: | ---: | ---: | ---: |
| A0 rocBLAS FP16 | 0.7139 | 2,868,713 | 150.40 | 1.00× |
| B0 current WMMA + pre-expanded FP16 fragment | 4.7511 | 431,055 | 22.60 | 6.65× slower |
| B3 current EXL3 mul1 | 6.6431 | 308,291 | 16.16 | 9.31× slower |

- A0 is a plain FP16 GEMM (no EXL3 epilogue). B0/B3 include the EXL3 affine /
  inverse Hadamard / `svh` epilogue.
- B0 reads pre-decoded raw FP16 tiles directly as the WMMA A fragment; it skips
  trellis bit extraction, mul1 hash, SAD, and the LDS tile round-trip.
- B0 and B3 outputs are **bit-identical** (`max_abs=0`) for the full M×N output.
- The current WMMA structure is **6.65× slower than rocBLAS before any decode**.
  The entire EXL3 decode adds only 1.40× on top of B0.
- Therefore the dominant gap is the WMMA tile structure / data supply / scheduling,
  not the EXL3 decode chain. Further decode micro-optimization has at most ~1.4×
  headroom in this kernel.
- Next: B1 (LDS layout) and B2 (hash/SAD) are optional for decomposing the 1.4×,
  but the larger opportunity is a large-tile WMMA GEMM (128×128 or larger,
  register blocking, K pipelining) or a dense GEMM with 128-row weight reuse.

## Build

```bash
hipcc -O3 -std=c++20 --offload-arch=gfx1201 \
  -I include \
  -I /opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/include \
  tests/benchmarks/exl3_isolation/exl3_isolation.hip \
  tests/benchmarks/exl3_isolation/main.cpp \
  -o /tmp/iso_bench \
  -L/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib \
  -lrocblas \
  -Wl,-rpath,/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib
HIP_VISIBLE_DEVICES=1 /tmp/iso_bench
```

## Files

- `exl3_isolation.hip` — production EXL3 kernel plus a `kDense` direct-A-fragment
  path and a trellis→dense-tile recon kernel.
- `main.cpp` — A0 rocBLAS, B0, B3, timing, correctness.
- `isa/stage_b0.s`, `isa/stage_b3.s` — generated gfx1201 ISA for the two kernels.
- `result.csv` — machine-readable results.
