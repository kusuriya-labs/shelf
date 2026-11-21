"""gfx1030 FP16 BC decode-attention tuning (rocm_py helper).

What this tunes (measured 2026-09-28, same-input A/B on D = Qwen3-8B 4bpw,
8192-prompt / 64-output, warmup 1 + 2 repeats; /work/runs/d-aot-{base,align,
rows,both}.json, input hashes verified identical by
/work/runs/aot-experiment-index.json):

    base 17.6 tok/s | alignment hints 42.8 | narrow rows 26.1 | both 47.2

The second verified shape (M, the num_kv_heads == 4 Qwen3 model) went
29.5 -> 56.3 tok/s on the same protocol; tuned-vs-baseline top-1 agreement
over the 1024-position manifest: D 1021/1024, M 1024/1024.

Two independent changes to the AOT-compiled flash-decoding kernels for one
narrowly validated shape family (bc_attn.BCAttn._configure calls in here):

1. Narrow head tile. BLOCK_H starts at its default 16 rows per program
   (max(16 // BLOCK_M, 1)); the GQA group is only 4 or 8 heads for the
   eligible shapes, so 8-12 masked-out rows rode in every tile. Replacing
   BLOCK_H with min(BLOCK_H, next_power_of_2(group_size)) shrinks the tile
   to the real group. A raw decode-kernel row sweep (/work/runs/
   decode-row-sweep.json) found no output difference from narrowing for
   the tested shapes, so this is wasted-work removal, not a math change.

2. Alignment hints. The AOT signatures gain ":16" (tt.divisibility, see
   bc_attn._compile_kernel) on the pointers the launch actually passes --
   q, k_cache, v_cache, out, partial_o, partial_ml -- and on split_len.
   Without it the AOT kernel loses vectorized global loads (~1.5x on these
   bandwidth-heavy decode kernels).

The alignment promise is checked, not assumed:

  * q / o / partial_o / partial_ml are this configure's g_tensor_cache
    torch.empty allocations (partial_* via get_bucketed, whose slice starts
    at element 0 of the bucket, so data_ptr is the allocation base). The
    C++ launch passes exactly these tensors' data_ptr() (attention.cpp:
    s.q / s.o / s.partial_o / s.partial_ml come from configure_slot's
    copies of the very same tensors), so verifying .data_ptr() % 16 here
    verifies the launched pointer.
  * k_cache / v_cache are the cache allocations handed to BCAttn's
    constructor (CacheLayer_fp16.k / .v, or the SWA state view); they too
    are launched by data_ptr() from the stored at::Tensor. Views can in
    principle carry an offset, so these are qualified per configure.
  * split_len is a runtime scalar, not a pointer: attention.cpp
    split_config computes CEIL_DIVIDE(CEIL_DIVIDE(bound, num_splits),
    block_n) * block_n, always a multiple of block_n (64 for head_dim
    128), hence divisible by 16.
  * block_table / cache_seqlens / the other scalar sizes are NOT hinted:
    their width changes with the generated length (block table growth) and
    num_splits is arbitrary, so no divisibility is promised or needed.

The fallbacks, in order of when they can fire: an unaligned cache pointer
is a genuine configuration possibility (the cache tensors can arrive as
views), so it is qualified up front and declines the tuning WHOLESALE:
original geometry, original signatures. The owned statics (q / o /
partial_o / partial_ml) are this configure's torch allocations and always
aligned in practice; they are asserted after allocation, before
configure_slot registers the optimized slot, and a violation RAISES rather
than launching: the kernels were compiled with the hints, so proceeding
would be the unsafe promise. Every successful prototype run had aligned
caches.

Eligibility (everything else keeps the original behavior, on every other
shape / device / build / arch, including CUDA): torch HIP build, the
device's actual gcnArchName == gfx1030 (colon suffix tolerated),
q_len == 1, head_dim == 128, num_q_heads == 32, num_kv_heads in (4, 8),
FP16 cache (k_bits == 0 and v_bits == 0), ordinary dense attention (not
QSA), and no output gate (gate_mode == 0: the verified Qwen3 models have
none, and the gate modes were never A/B'ed). Host-side geometry flows
through the narrowed BLOCK_H too (see bc_attn._configure), and for these
shapes the narrowed tile keeps h_blocks == 1 -- identical to the original
block_h = 16 -- so the program count stays consistent with the C++ grid
derivation, which re-computes h_blocks from the ORIGINAL formula.

EXL3_ROCM_GQA_TUNE=0 opts out (default on for eligible shapes; set it
before model load / configure, switches are read per configure).

Prototype: /path/to/rocm-exl3-data/bench_aot_tune.py wrapped
bc_attn._compile_kernel and rewrote the constexprs on the way in -- correct
results at 47.2 tok/s but the host scratch buffers stayed sized for the
untuned 16-row tile. This helper is the same tuning expressed at
_configure, so buffer allocation matches the narrowed kernel constants.
"""

from __future__ import annotations

ENV_SWITCH = "EXL3_ROCM_GQA_TUNE"

# Signature entries that get the ":16" divisibility promise (see the module
# docstring for why each one is safe). Everything else in the signature --
# block_table, cache_seqlens, k_scales/v_scales/h32/sinks, num_splits,
# num_pages_per_seq -- stays unhinted.
SPLIT_ALIGNED_ARGS = ("q", "k_cache", "v_cache", "out", "partial_o",
                      "partial_ml", "split_len")
COMBINE_ALIGNED_ARGS = ("partial_o", "partial_ml", "out")

_ARCH = "gfx1030"

_is_hip: bool | None = None
_arch_by_index: dict[int, str] = {}


def next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def arch_supported(gcn_arch_name: str) -> bool:
    """gfx1030 exactly (the V620 the tuning was measured on). A colon suffix
    (e.g. feature flags in some builds' gcnArchName) is tolerated; family
    siblings (gfx1031/32/34/35) are NOT: nothing has been measured there."""
    return str(gcn_arch_name).split(":", 1)[0] == _ARCH


def env_enabled(environ = None) -> bool:
    """EXL3_ROCM_GQA_TUNE switch, default on; semantics as in rocm_py._env_on."""
    import os
    if environ is None:
        environ = os.environ
    v = environ.get(ENV_SWITCH)
    if v is None:
        return True
    return v.strip() not in ("", "0", "false", "False")


def is_hip_build() -> bool:
    global _is_hip
    if _is_hip is None:
        try:
            import torch
            _is_hip = getattr(torch.version, "hip", None) is not None
        except Exception:
            _is_hip = False
    return _is_hip


def device_gcn_arch(device) -> str | None:
    """Cached per-device gcnArchName. Returns None for anything this cannot
    read (cpu device string, missing CUDA context, driver error): the caller
    treats that as 'not gfx1030'."""
    import torch
    if isinstance(device, str):
        device = torch.device(device)
    if getattr(device, "type", None) != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _arch_by_index:
        try:
            _arch_by_index[index] = str(
                getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))
        except Exception:
            _arch_by_index[index] = ""
    return _arch_by_index[index] or None


def decode_tune_eligible(*, is_hip: bool, gcn_arch_name: str, env_on: bool,
                         q_len: int, head_dim: int, num_q_heads: int,
                         num_kv_heads: int, k_bits: int, v_bits: int,
                         gate_mode: int, qsa: bool) -> bool:
    """Pure shape/device/env predicate -- the CPU-testable core. All inputs
    explicit so a test on any host can exercise the real decision table."""
    if not env_on or not is_hip or qsa:
        return False
    if q_len != 1 or head_dim != 128:
        return False
    if num_q_heads != 32 or num_kv_heads not in (4, 8):
        return False
    if k_bits != 0 or v_bits != 0:
        return False
    if gate_mode != 0:
        return False
    return arch_supported(gcn_arch_name)


def decode_tune_enabled(device, *, q_len: int, head_dim: int, num_q_heads: int,
                        num_kv_heads: int, k_bits: int, v_bits: int,
                        gate_mode: int, qsa: bool) -> bool:
    """One-call check for bc_attn._configure: reads the env switch, the torch
    build and the device's actual gcnArchName, then delegates. Nothing here
    raises: an unprobeable device simply is not eligible."""
    if not (env_enabled() and is_hip_build()):
        return False
    return decode_tune_eligible(
        is_hip = True, gcn_arch_name = device_gcn_arch(device) or "",
        env_on = True, q_len = q_len, head_dim = head_dim,
        num_q_heads = num_q_heads, num_kv_heads = num_kv_heads,
        k_bits = k_bits, v_bits = v_bits, gate_mode = gate_mode, qsa = qsa)


def narrowed_block_h(block_h: int, group_size: int) -> int:
    """Rows per program cover the GQA group exactly (rounded up to a pow2
    tile), never more than the caller's original block_h. h_blocks =
    cdiv(group_size, result) stays 1 for the eligible shapes, so the grid
    matches the untouched C++ derivation; block_rows and the scratch sizes
    computed from the returned value shrink with the tile."""
    return min(block_h, next_power_of_2(group_size))


def pointers_aligned(ptrs, align: int = 16) -> bool:
    """True only when every pointer is divisible by align; an empty list is
    False (there is nothing to promise -- callers pass the exact pointer set
    the tuned signature marks)."""
    if not ptrs:
        return False
    return all(int(p) % align == 0 for p in ptrs)


def with_alignment_hints(signature: dict, names, suffix: str = ":16") -> dict:
    """Copy of the AOT signature with the divisibility suffix on the listed
    string entries (pointer types and plain scalars alike; constexpr entries
    and unlisted names pass through). Idempotent: already-marked entries are
    not double-suffixed."""
    out = dict(signature)
    for n in names:
        ty = out.get(n)
        if isinstance(ty, str) and ty != "constexpr" and not ty.endswith(suffix):
            out[n] = ty + suffix
    return out
