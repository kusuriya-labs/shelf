"""ROCm/RDNA Python-side overrides for exllamav3.

The C++ side keeps upstream sources byte-identical and puts every ROCm-specific
change in ``exllamav3_ext/rocm/`` as a sibling file (see that directory's
README). This package is the same idea for Python: instead of editing upstream
modules in place, the divergences live here and are applied as monkeypatches at
import time, from a single hook at the end of ``exllamav3/__init__.py``.

Why not edit the modules directly? The original ROCm fork did, across five
files, and every upstream rebase then had to re-derive which edits were ROCm
workarounds and which were upstream changes. Keeping them here means
``git diff`` against upstream stays empty for the shared code, and each
divergence carries its own reason and its own off switch.

A patch that cannot be applied reports "!! FAILED" in describe() rather than
being swallowed. A bisect handle that silently does nothing is worse than no
handle at all -- it makes a live kernel look disabled.

Nothing here runs unless ``torch.version.hip`` is set, so a CUDA build imports
this module and does nothing.

Environment switches (all default to the safe value for this backend):

  EXL3_ROCM_PATCH=0        disable every patch below
  EXL3_ROCM_MGEMM=0        distrust exl3_mgemm on RDNA: disables MultiLinear
                           fusion *and* the bsz-1 MoE mgemm routes. Default on
                           since 2026-08-07 (see the note at the patch)
  EXL3_ROCM_MOE_DISABLE=1  route block-sparse MoE through the dense per-expert
                           path instead of the fused kernel (NOT advised -- see
                           the note at the patch itself)
  EXL3_ROCM_MLP_RANGE_BALANCE=0  on gfx1030, do not rescale gated-MLP up/down
                           svh metadata (up /= 8, down *= 8) at forward entry
                           to keep silu(g)*u inside fp16 range (d-reference:
                           3/8 cases -> Inf; the Model.load-wrapping prototype
                           restored full finiteness at 94.63% top-1 vs native
                           BF16 -- GPU verification of this guard is pending
                           via rocm_tools/rdna2/mlp_range_balance_probe.py)
  EXL3_ROCM_RDNA4_FUSED_MOE=1  on gfx120x, do not steer MoE off the fused
                           kernel (whose WMMA traps on gfx12); for a future
                           gfx12 WMMA port

  Bisect handles -- slow, for localising a numerics fault, never to leave on:

  EXL3_ROCM_MOE_TORCH=1    MoE expert compute in pure torch
  EXL3_ROCM_ROUTING_TORCH=1  expert routing in pure torch (routing_ds3)
  EXL3_ROCM_FORCE_TORCH=1  every EXL3 Linear via reconstruct + at::mm, taking
                           exl3_gemm and exl3_gemv out of the model

  Added at the v1.5.0 sync (2026-09-20). Each keeps a v1.4.4-validated path as
  the default and makes the new upstream path opt-in until it has been run on
  RDNA:

  EXL3_ROCM_MOE_BSZN=1     let bsz <= MAX_BSZN MoE decode take upstream's
                           BC_BlockSparseMLP.run_bszN route. On ROCm that route
                           is the unported exl3_moe_coop kernel and raises;
                           default off steers it (see the next switch)
  EXL3_ROCM_MOE_MGEMM_ROUTE=0  steer bsz <= MAX_BSZN MoE decode to the fused
                           exl3_moe kernel (a 16-row tile GEMM padding one
                           useful row: ~half the decode speed). Default on =
                           the restored v1.4.4 per-token exl3_mgemm route,
                           which lands on the mgemv fast path
  EXL3_ROCM_MOE_MGEMM_MAX_ROWS=N  opt-in decode-row cap of the mgemm route
                           (default 8, valid 8..24, e.g. 20 for batch 4 x
                           draft 4 MTP verify): raises the PYTHON
                           block_sparse_mlp.MAX_BSZN once at patch time,
                           before any load, so rows within the cap stay on
                           the per-token exl3_mgemm route (scratch sized at
                           load accordingly) instead of the fused exl3_moe
                           fallback. Applies ONLY under the default steer
                           above; other steers keep their semantics and
                           report the knob ignored. mlp.MAX_BSZN and the
                           compiled native cap stay 8. An invalid value
                           aborts startup (ValueError at import), never
                           log-and-fallthrough into the native coop stub.
                           Diagnostic: rocm_tools/rdna2/moe_rows_probe.py
  EXL3_ROCM_MOE_MULTI_TOKEN=0  opt out of the experimental multi-token route
                           (default enabled on native-compatible gfx10/gfx11);
                           the r3 single-layer
                           probe and a 96-sample actual-model TP2/MTP gate
                           validate the measured batch-1 route, while general
                           batch shapes remain unvalidated
                           (rocm_tools/rdna2/moe_multitoken_probe.py):
                           decode rows 2..configured cap (8..24) take
                           row-aligned chunks of ONE slot-major exl3_mgemm
                           triple (gate/up/down with num_tokens=chunk rows)
                           instead of the per-token loop. Each chunk keeps
                           R*top_k <= 128 slots, quant K in 1..8,
                           fast-path-aligned dims (k/n % 128), contiguous
                           half y / int64 selected / half routing_weights;
                           R == 1 and anything unsupported keep the UNCHANGED
                           row loop. Native-compatible gfx10/gfx11 only
                           (arch resolved through the
                           cached per-device probe, never re-queried per
                           decode step). Expert-range shards pre-rebase on
                           GPU -- torch.where(in-range, sel - min_expert,
                           -1), all R*top_k slots kept in position order --
                           and pass min/max = -1/-1, because the barrier-free
                           mgemv fast path declines num_tokens>1 with
                           min_index>=0 and would otherwise fall to the
                           cooperative kernel. The fast path's grouped reduce
                           has NO -1 guard: its fallback kernel sums every
                           slot row of C (exl3_mgemv_reduce_kernel) and its
                           fused epilogue's arrival counter never completes
                           when masked-slot blocks skip their arrival, then
                           never self-resets (device-wide poisoning of later
                           weighted calls). Older binaries therefore zero the
                           down-proj slot scratch first -- inactive
                           slots contribute exact +0.0 to the fp32 chain;
                           the r3 probe measured the masked route (maximum
                           candidate relative-L2 9.644e-5 and scaled absolute
                           error 1.1775e-4 across its 15 full/TP-half cases),
                           and the actual-model gate covered 96 live-input
                           cases (48 per rank; maximum relative-L2 3.9123e-4,
                           scaled absolute error 6.6285e-4, all finite with
                           restored env/output); measured batch-1 TP2/MTP
                           throughput improved +8.055% observed / +11.236%
                           engine, while general batch shapes remain unvalidated --
                           Older native binaries zero down scratch and run
                           their weighted launch with EXL3_GEMV_FUSE_OUT=0,
                           set and restored around that call; binaries with
                           EXL3_MGEMV_MASKED_REDUCE_SUPPORTED use their native
                           masked reduction without the process toggle.
                           Unsharded modules can never
                           produce -1 picks and keep the fused form. Gateless
                           relu2 experts use up/activation/down when their
                           native layouts pass the same guards. Arbitrary
                           concurrent native callers are unsupported; TP2
                           rank processes have separate environments.
  EXL3_ROCM_QKV_SLICE=1    enable the one-launch sliced Q/K/V bundle
                           (SlicedMultiLinear, exl3_mgemm sliced mode). Ported
                           into the WMMA kernels, unvalidated on RDNA
  EXL3_ROCM_BATCH_RECON=1  enable the batched expert-reconstruct prefill tier
                           (reconstruct_*_batch + hgemm_batched). Ported
                           mechanically, unvalidated on RDNA
  EXL3_ROCM_MOE_MTILE=1    let Python split fused-MoE launches into 16/32/64-row
                           tiers. Pointless on RDNA, which only builds the
                           16-row instance and runs every tier through it

These are bisect handles, not permanent policy -- turn one on, run a prompt, see
whether the output degrades. Each one's justification is a measurement recorded
at the patch, not an inherited assumption; a guard whose reason has gone stale
is a guard that should be retested and deleted.
"""

from __future__ import annotations
import os
import threading


def _env_on(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip() not in ("", "0", "false", "False")


# This cap sizes Python MoE scratch and selects the ROCm per-token proxy.
# It does not change the compiled native or shared-MLP row limits.
MOE_MGEMM_MAX_ROWS_DEFAULT = 24
MOE_MGEMM_MAX_ROWS_RANGE = (8, 24)


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    """Parse an integer setting and reject invalid values before patching."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    txt = raw.strip()
    try:
        val = int(txt)
    except ValueError:
        raise ValueError(
            f"{name}={txt!r}: expected a whole number in {lo}..{hi} "
            f"(or unset for the default {default})") from None
    if not lo <= val <= hi:
        raise ValueError(
            f"{name}={txt}: out of range, expected a whole number in {lo}..{hi} "
            f"(default {default})")
    return val


# ----------------------------------------------------------------------
# EXL3_ROCM_MOE_MULTI_TOKEN -- experimental slot-major multi-token route
# ----------------------------------------------------------------------
# One (gate, up, down) triple of exl3_mgemm calls carries every decode row
# (num_tokens = R) instead of R independent num_tokens = 1 calls. Lives at
# module level so the probe (rocm_tools/rdna2/moe_multitoken_probe.py) and
# the CPU contract tests drive the same code the model decode runs.
MOE_MT_ENV = "EXL3_ROCM_MOE_MULTI_TOKEN"
MOE_MT_FUSE_ENV = "EXL3_GEMV_FUSE_OUT"
MOE_MT_ROWS_MIN = 2
# Kept as the CPU-contract fallback; apply() replaces _MOE_MT["max_rows"] with
# EXL3_ROCM_MOE_MGEMM_MAX_ROWS (8..24) for the loaded native route.
MOE_MT_ROWS_MAX = 5
MOE_MT_ROWS_LIMIT = 24
MOE_MT_SLOTS_MAX = 128   # MAX_INDICES, the cooperative/fast bound; A slot-major

# Wiring done by apply() under the default mgemm steer; tests inject fakes.
# require_cuda is True for the real route (the kernels take device pointers);
# CPU tests clear it to exercise the gating and slot math on CPU tensors.
_MOE_MT: dict = {
    "on": False,
    "reason": f"{MOE_MT_ENV} not requested",
    "torch": None,
    "ext": None,
    "tcache": None,
    "arch_ok": None,        # cached per-device gfx1030 probe, resolved at load
    "arch_cache": {},       # device-index -> bool, populated once at apply()
    "max_rows": MOE_MT_ROWS_MAX,
    "require_cuda": True,
}

# EXL3_GEMV_FUSE_OUT is read by the native dispatcher while the launch is
# submitted.  Serializing this short process-environment mutation prevents two
# Python callers using this experimental route from changing it between the
# assignment and the launch.  The candidate still carries an explicit
# single-thread/process limitation: callers outside this lock can race an env
# read in native code, and graph capture has its own fixed environment.
_MOE_MT_FUSE_LOCK = threading.Lock()


def moe_multi_token_scratch_clear(mod) -> None:
    """Release candidate staging owned by one module, if it has any.

    The route keeps a bounded 128-slot staging slab on each loaded module so
    layers and target/draft modules cannot alias a global cache entry.  The
    patched BlockSparseMLP.unload calls this helper; it is public so probes and
    alternative loaders can make the same lifetime boundary explicit.
    """
    scratch = getattr(mod, "_rocm_moe_mt_scratch", None)
    if scratch is not None:
        scratch.clear()
        try:
            delattr(mod, "_rocm_moe_mt_scratch")
        except AttributeError:
            pass


def _moe_mt_scratch(mod, y, cfg=None, slots=0):
    """Return bounded, module-owned staging tensors for one hidden width.

    Most loaded modules already have enough ``experts_cfg`` capacity because
    the normal bszN row cap is larger than this experiment's five rows.  A
    loader with only top-k rows is still valid: when any active slot would
    exceed that capacity, allocate one fixed-capacity native scratch bundle on
    the module and use it for all five arguments.
    """
    t = _MOE_MT["torch"]
    hi_ = int(y.shape[1])
    dev = y.device
    need_native = bool(cfg is not None and any(
        b.shape[0] < slots for b in
        (cfg.yh, cfg.interm_g, cfg.interm_u, cfg.interm_a, cfg.out_d)))
    cfg_sig = None
    if cfg is not None:
        cfg_sig = (
            tuple(cfg.yh.shape), str(cfg.yh.dtype),
            tuple(cfg.interm_g.shape), str(cfg.interm_g.dtype),
            tuple(cfg.interm_u.shape), str(cfg.interm_u.dtype),
            tuple(cfg.interm_a.shape), str(cfg.interm_a.dtype),
            tuple(cfg.out_d.shape), str(cfg.out_d.dtype),
        )
    native_sig = cfg_sig if need_native else None
    scratch = getattr(mod, "_rocm_moe_mt_scratch", None)
    def add_native_bundle(dst):
        # Keep the base object and its hidden/index buffers when promoting
        # from an R that fit cfg capacity to a larger R.  Promotion should
        # allocate the five native tensors once, without needlessly replacing
        # reusable slot staging.
        dst["native_sig"] = native_sig
        dst["native"] = {
            "yh": t.empty((MOE_MT_SLOTS_MAX, 1, cfg.yh.shape[-1]),
                           dtype=cfg.yh.dtype, device=dev),
            "interm_g": t.empty((MOE_MT_SLOTS_MAX, 1, cfg.interm_g.shape[-1]),
                                 dtype=cfg.interm_g.dtype, device=dev),
            "interm_u": t.empty((MOE_MT_SLOTS_MAX, 1, cfg.interm_u.shape[-1]),
                                 dtype=cfg.interm_u.dtype, device=dev),
            "interm_a": t.empty((MOE_MT_SLOTS_MAX, 1, cfg.interm_a.shape[-1]),
                                 dtype=cfg.interm_a.dtype, device=dev),
            "out_d": t.empty((MOE_MT_SLOTS_MAX, 1, cfg.out_d.shape[-1]),
                              dtype=cfg.out_d.dtype, device=dev),
        }
    if scratch is not None:
        base_match = (scratch.get("device") == dev and scratch.get("hidden") == hi_
                      and scratch.get("torch") is t
                      and scratch.get("cfg_sig") == cfg_sig)
        if base_match:
            # A route may first run with R*K inside cfg capacity, then grow
            # beyond it on the same module. Promote exactly once; retain an
            # already-promoted bundle even for smaller later R.
            if need_native and "native" not in scratch:
                add_native_bundle(scratch)
            return scratch
        moe_multi_token_scratch_clear(mod)

    # One fixed-capacity hidden slab plus two tiny vectors per loaded module.
    # This is deliberately separate from g_tensor_cache: that cache is keyed
    # by shape/tag and would alias same-shaped layers or target/draft modules.
    scratch = {
        "device": dev,
        "hidden": hi_,
        "torch": t,
        "cfg_sig": cfg_sig,
        "hidden_slots": t.empty((MOE_MT_SLOTS_MAX, hi_), dtype=t.half, device=dev),
        "indices": t.empty((MOE_MT_SLOTS_MAX,), dtype=t.long, device=dev),
        "mask": t.empty((MOE_MT_SLOTS_MAX,), dtype=t.bool, device=dev),
    }
    if need_native:
        # Keep this bounded at the same 128-slot experiment cap.  The bundle
        # is only materialized for loaders whose ordinary buffers are shorter
        # than the active route; all views passed to native are narrowed to
        # exactly ``slots`` below.
        add_native_bundle(scratch)
    setattr(mod, "_rocm_moe_mt_scratch", scratch)
    return scratch


def moe_multi_token_status() -> dict:
    """Snapshot of the candidate route's wiring, for probes and describe()."""
    return {
        "on": bool(_MOE_MT["on"]),
        "reason": str(_MOE_MT["reason"]),
        "rows": (MOE_MT_ROWS_MIN, int(_MOE_MT.get("max_rows", MOE_MT_ROWS_MAX))),
        "slots_max": MOE_MT_SLOTS_MAX,
        "require_cuda": bool(_MOE_MT["require_cuda"]),
        "masked_reduce_capability": bool(
            getattr(_MOE_MT.get("ext"), "EXL3_MGEMV_MASKED_REDUCE_SUPPORTED", False)),
    }


def moe_multi_token_supported(mod, y, selected_experts, routing_weights):
    """(bool, why-not) gate for the slot-major multi-token route.

    Cheap, host-side checks only -- no .item()/nonzero/sort, no GPU-property
    query (the arch probe caches per device index). Every refusal leaves the
    unchanged per-token row loop in charge."""
    t = _MOE_MT["torch"]
    if t is None or _MOE_MT["ext"] is None or _MOE_MT["tcache"] is None:
        return False, "route not wired (apply() mgemm steer inactive)"
    # Do not report a candidate run when the native GEMV kill switch would
    # send these launches to cooperative GEMM.  The unchanged row loop owns
    # that configuration instead.
    if not _env_on("EXL3_MGEMV", True):
        return False, "EXL3_MGEMV disabled"
    if y.dim() != 2 or selected_experts.dim() != 2 or routing_weights.dim() != 2:
        return False, "expected 2-D y / selected_experts / routing_weights"
    r, k = selected_experts.shape
    if y.shape[0] != r:
        return False, f"y rows {y.shape[0]} != selected rows {r}"
    if tuple(routing_weights.shape) != (r, k):
        return False, (f"routing weights shape {tuple(routing_weights.shape)} "
                       f"!= selected shape {(r, k)}")
    max_rows = int(_MOE_MT.get("max_rows", MOE_MT_ROWS_MAX))
    if not MOE_MT_ROWS_MIN <= r <= max_rows:
        return False, f"rows {r} outside {MOE_MT_ROWS_MIN}..{max_rows}"
    if k < 1 or r * k > MOE_MT_SLOTS_MAX:
        # Larger row batches are split into row-aligned chunks below; the
        # whole call itself may exceed MAX_INDICES while each native launch
        # remains within its 128-slot bound.
        if k < 1 or MOE_MT_SLOTS_MAX // k < MOE_MT_ROWS_MIN:
            return False, f"top_k={k} cannot form a multi-token <= {MOE_MT_SLOTS_MAX}-slot chunk"
    cfg = getattr(mod, "experts_cfg", None)
    if cfg is None:
        return False, "experts_cfg unloaded"
    gated = bool(getattr(mod, "gated", False))
    mg, mu, md = mod.multi_gate, mod.multi_up, mod.multi_down
    if mu is None or md is None or (gated and mg is None):
        return False, "fused MultiLinear tables absent"
    mats = ([ ("gate", mg) ] if gated else []) + [("up", mu), ("down", md)]
    for nm, m in mats:
        if not 1 <= int(m.K) <= 8:      # the fast path's template switch
            return False, f"{nm} K={m.K} outside 1..8"
    if y.dtype != t.half or selected_experts.dtype != t.long \
            or routing_weights.dtype != t.half:
        return False, "unexpected dtypes (want half y, int64 selected, half weights)"
    if selected_experts.device != y.device or routing_weights.device != y.device \
            or cfg.out_d.device != y.device:
        return False, "input/cfg device mismatch"
    if _MOE_MT["require_cuda"]:
        if not y.is_cuda:
            return False, "non-CUDA tensors"
        device_index = y.device.index
        if device_index is None:
            device_index = t.cuda.current_device()
        arch_cache = _MOE_MT.get("arch_cache", {})
        if device_index not in arch_cache:
            return False, "arch probe not wired"
        if not arch_cache[device_index]:
            return False, "device is outside native gfx10/gfx11 envelope"
    hi_ = y.shape[1]
    if cfg.yh.shape[-1] != hi_ or hi_ % 128:
        return False, f"hidden width {hi_} != padded gate/up k or not %% 128"
    i_ = cfg.interm_a.shape[-1]
    if i_ % 128 or cfg.interm_g.shape[-1] != i_ or cfg.interm_u.shape[-1] != i_:
        return False, f"intermediate width {i_} not %% 128 or gate/up disagree"
    ho = cfg.out_d.shape[-1]
    if ho % 128 or cfg.out_bszn.shape[-1] > ho:
        return False, f"down output width {ho} not %% 128 / out_bszn wider"
    if (cfg.yh.dim() != 3 or cfg.yh.shape[1] != 1
            or cfg.interm_g.dim() != 3 or cfg.interm_g.shape[1] != 1
            or cfg.interm_u.dim() != 3 or cfg.interm_u.shape[1] != 1
            or cfg.interm_a.dim() != 3 or cfg.interm_a.shape[1] != 1
            or cfg.out_d.dim() != 3 or cfg.out_d.shape[1] != 1):
        return False, "native scratch must have [rows,1,width] shape"
    if not all(b.is_contiguous() for b in
               (cfg.yh, cfg.interm_g, cfg.interm_u, cfg.interm_a, cfg.out_d)):
        return False, "native scratch must be contiguous"
    if cfg.out_bszn.dim() != 2 or not cfg.out_bszn.is_contiguous():
        return False, "out_bszn must be a contiguous [rows,width] tensor"
    if cfg.yh.dtype != t.half or cfg.interm_a.dtype != t.half:
        return False, "native A/A_had and activation scratch must be half"
    if (cfg.interm_g.dtype not in (t.half, t.float)
            or cfg.interm_u.dtype != cfg.interm_g.dtype
            or cfg.out_d.dtype not in (t.half, t.float)):
        return False, "unsupported native scratch dtype"
    if cfg.interm_a.data_ptr() in (cfg.interm_g.data_ptr(), cfg.interm_u.data_ptr()):
        return False, "activation scratch aliases gate/up output"
    slots = r * k
    # The ordinary loader may provide only top-k rows.  _moe_mt_scratch()
    # allocates one bounded module-owned native bundle for that case, so a
    # short cfg buffer is a reason to stage, not a reason to silently poison
    # the grouped stride.
    if MOE_MT_SLOTS_MAX // k < MOE_MT_ROWS_MIN:
        return False, f"top_k={k} cannot form a multi-token <= {MOE_MT_SLOTS_MAX}-slot chunk"
    if cfg.out_bszn.shape[0] < r:
        return False, f"out_bszn scratch has {cfg.out_bszn.shape[0]} rows < {r}"
    mine, maxe = cfg.min_expert, cfg.max_expert
    if (mine is None) != (maxe is None):
        return False, "expert range must provide both min_expert and max_expert"
    if mine is not None and ((mine < 0) != (maxe < 0)):
        return False, f"malformed expert range [{mine},{maxe})"
    if mine is not None and mine >= 0:
        if not mine < maxe:
            return False, f"degenerate expert range [{mine},{maxe})"
        for nm, m in mats:
            if m.ptrs_trellis.shape[0] < maxe - mine:
                return False, (f"{nm} pointer table holds {m.ptrs_trellis.shape[0]} "
                               f"entries < shard width {maxe - mine}")
    return True, ""


def _moe_mt_run(mod, y, selected_experts, routing_weights, out_row_offset=0):
    """Run one row-aligned slot-major launch triple.

    Slots are (row, pick) flattened token-major: slot j = row j//k, pick j%k,
    so the native grouped reduce (num_tokens=r, stride = slots/r = k) reads
    exactly this layout, A is addressed per slot when bszm_in > 1, and the
    weights are addressed by original position (min_index = -1 => no packing).
    """
    t = _MOE_MT["torch"]
    ext = _MOE_MT["ext"]
    cfg = mod.experts_cfg
    r, k = selected_experts.shape
    slots = r * k
    hi_ = y.shape[1]

    # Bounded reusable staging is owned by this module.  Keeping the full
    # capacity in one object avoids same-shaped layers (or target/draft
    # modules) accidentally sharing a global cache entry; the narrow views
    # below are exactly the launch dimensions seen by native code.
    scratch = _moe_mt_scratch(mod, y, cfg, slots)
    a_slots = scratch["hidden_slots"].narrow(0, 0, slots).view(slots, 1, hi_)
    idx_buf = scratch["indices"].narrow(0, 0, slots)
    mask_buf = scratch["mask"].narrow(0, 0, slots)

    idx_buf.copy_(selected_experts.reshape(-1))
    mine, maxe = cfg.min_expert, cfg.max_expert
    sharded = mine is not None and maxe is not None and mine >= 0
    masked_reduce_capability = bool(
        getattr(ext, "EXL3_MGEMV_MASKED_REDUCE_SUPPORTED", False))
    if sharded:
        # Position-preserving LOCAL indices: in-range picks rebase to the
        # shard's pointer tables, out-of-range picks become -1 (skip). All
        # r*k slots and their order -- hence every token's fixed slot run --
        # stay intact; no compaction, no dedup, no host sync.
        idx_buf.sub_(mine)
        t.ge(idx_buf, maxe - mine, out=mask_buf)
        idx_buf.masked_fill_(mask_buf, -1)
        t.lt(idx_buf, 0, out=mask_buf)
        idx_buf.masked_fill_(mask_buf, -1)
    # mgemv indexes A by SLOT j when bszm_in > 1: repeat each row's hidden k times.
    a_slots.view(r, k, hi_).copy_(y.reshape(r, 1, hi_))
    idx = idx_buf.view(1, slots)
    w = routing_weights.reshape(1, slots)

    gated = bool(getattr(mod, "gated", False))
    mg, mu, md = mod.multi_gate, mod.multi_up, mod.multi_down
    # Every native argument whose outer dimension contributes to bszm is
    # sliced to the active slot count.  Passing the full per-layer capacity
    # would make grouped reduction use that capacity as its token stride and
    # could read past the position-preserving index vector.
    native = scratch.get("native")
    if native is None:
        yh = cfg.yh.narrow(0, 0, slots)
        interm_g = cfg.interm_g.narrow(0, 0, slots)
        interm_u = cfg.interm_u.narrow(0, 0, slots)
        interm_a = cfg.interm_a.narrow(0, 0, slots)
        out_d = cfg.out_d.narrow(0, 0, slots)
    else:
        yh = native["yh"].narrow(0, 0, slots)
        interm_g = native["interm_g"].narrow(0, 0, slots)
        interm_u = native["interm_u"].narrow(0, 0, slots)
        interm_a = native["interm_a"].narrow(0, 0, slots)
        out_d = native["out_d"].narrow(0, 0, slots)
    out_bszn = cfg.out_bszn.narrow(0, out_row_offset, r)

    # Native mgemv skips -1 slots without writing their C rows.  Clear both
    # gate/up outputs before those launches so activation sees zeros for an
    # inactive pick (including an all-invalid row); clear down rows below
    # before the weighted grouped reduction for the same reason.
    if sharded:
        interm_g.zero_()
        interm_u.zero_()
    if gated:
        ext.exl3_mgemm(
            a_slots, mg.ptrs_trellis, interm_g, mg.ptrs_suh, yh, mg.ptrs_svh,
            idx, None, mg.K, -1, mg.mcg, mg.mul1, -1, -1, 0, r, None, None)
    ext.exl3_mgemm(
        a_slots, mu.ptrs_trellis, interm_u, mu.ptrs_suh, yh, mu.ptrs_svh,
        idx, None, mu.K, -1, mu.mcg, mu.mul1, -1, -1, 0, r, None, None)
    if sharded:
        interm_a.zero_()
    mod.activation_fn_call(interm_g if gated else interm_u, interm_u, interm_a, mod.act_limit)

    if sharded and not masked_reduce_capability:
        # The fast path's grouped reduce has no -1 guard: the separate
        # exl3_mgemv_reduce_kernel sums every slot row of C, and the fused
        # epilogue's per-segment arrival counter never completes when masked
        # slots skip their arrival -- and then never self-resets, poisoning
        # later weighted calls device-wide. Zero the slot rows so masked
        # slots sum an exact +0.0 -- whether the unguarded fallback sum then
        # matches the cooperative kernel's guarded result (bit or within
        # rounding) is what the probe measures, not an assumption here --
        # and run this one weighted launch in the non-fused
        # (dot + rotate + reduce) form. The env is re-read per call by the
        # ext; gate/up keep the fused form because they carry no weights and
        # no reduction. A_had is the gate buffer, free after the activation
        # and never aliasing A = interm_a (the row loop's note).
        # NOTE: the r3 single-layer probe and 96-sample actual-model TP2/MTP
        # gate covered this zeroed-slot sum (actual maximum relative-L2
        # 3.9123e-4; scaled absolute error 6.6285e-4; all finite with env and
        # output restoration). The measured batch-1 route is supported as an
        # opt-in; general batch shapes remain unvalidated.
        # The EXL3_GEMV_FUSE_OUT toggle below is process-global and is an
        # experimental limitation: candidate callers are serialized and the
        # value is restored on every path, exception included, but arbitrary
        # concurrent native callers remain unsupported. TP2 rank processes
        # have separate environments.
        out_d.zero_()
        with _MOE_MT_FUSE_LOCK:
            prev = os.environ.get(MOE_MT_FUSE_ENV)
            os.environ[MOE_MT_FUSE_ENV] = "0"
            try:
                ext.exl3_mgemm(
                    interm_a, md.ptrs_trellis, out_d, md.ptrs_suh,
                    interm_g if gated else interm_u, md.ptrs_svh,
                    idx, w, md.K, -1, md.mcg, md.mul1, -1, -1, 0, r, None, None)
            finally:
                if prev is None:
                    os.environ.pop(MOE_MT_FUSE_ENV, None)
                else:
                    os.environ[MOE_MT_FUSE_ENV] = prev
    else:
        # Unsharded, or a native binary advertising masked weighted reduction:
        # keep the ambient fused form and let native skip inactive slots.
        ext.exl3_mgemm(
            interm_a, md.ptrs_trellis, out_d, md.ptrs_suh,
            interm_g if gated else interm_u, md.ptrs_svh,
            idx, w, md.K, -1, md.mcg, md.mul1, -1, -1, 0, r, None, None)

    width = out_bszn.shape[-1]
    out_bszn.copy_(out_d.narrow(0, 0, r).squeeze(1)[:, :width])


def _moe_mt_chunk_sizes(rows: int, chunk_rows: int) -> list[int]:
    """Split rows without emitting a one-row tail when multi-token is viable."""
    out = []
    left = rows
    while left:
        take = min(left, chunk_rows)
        rem = left - take
        if rem == 1 and take > MOE_MT_ROWS_MIN:
            take -= 1
            rem += 1
        elif rem == 1:
            return []
        out.append(take)
        left = rem
    return out


def moe_multi_token_step(mod, y, selected_experts, routing_weights) -> bool:
    """Run the candidate route if wired and supported; True iff it ran.

    Never partially executes: a refusal runs no kernel and leaves the caller
    to the unchanged row loop."""
    if not _MOE_MT["on"]:
        return False
    ok, _why = moe_multi_token_supported(mod, y, selected_experts, routing_weights)
    if not ok:
        return False
    k = int(selected_experts.shape[1])
    chunk_rows = MOE_MT_SLOTS_MAX // k
    if chunk_rows < MOE_MT_ROWS_MIN:
        return False
    chunks = _moe_mt_chunk_sizes(int(y.shape[0]), chunk_rows)
    if not chunks:
        return False
    offset = 0
    for take in chunks:
        _moe_mt_run(mod, y.narrow(0, offset, take),
                    selected_experts.narrow(0, offset, take),
                    routing_weights.narrow(0, offset, take), offset)
        offset += take
    return True


def moe_mgemm_bszN(mod, y, selected_experts, routing_weights):
    """The bsz<=cap MoE decode route: multi-token candidate when opted in and
    supported, otherwise -- including every R == 1 call -- the unchanged
    v1.4.4 per-token exl3_mgemm loop."""
    if moe_multi_token_step(mod, y, selected_experts, routing_weights):
        return
    _moe_mgemm_rowloop(mod, y, selected_experts, routing_weights)


def _moe_mgemm_rowloop(mod, y, selected_experts, routing_weights):
    # (Moved verbatim from the apply() closure so the multi-token candidate and
    # this fallback share one module; the body is the v1.4.4 per-token route.)
    ext = _MOE_MT["ext"]
    if ext is None:
        raise RuntimeError("rocm_py MoE mgemm route invoked before apply() wired it")
    cfg = mod.experts_cfg
    bsz = y.shape[0]
    mine, maxe = cfg.min_expert, cfg.max_expert
    A = y.unsqueeze(1).unsqueeze(1)          # (bsz, 1, 1, Hi)
    sel = selected_experts.unsqueeze(1)      # (bsz, 1, top_k)
    w = routing_weights.unsqueeze(1)         # (bsz, 1, top_k)
    width = cfg.out_bszn.shape[-1]
    top_k = selected_experts.shape[1]
    g_slots = cfg.interm_g.narrow(0, 0, top_k)
    u_slots = cfg.interm_u.narrow(0, 0, top_k)
    a_slots = cfg.interm_a.narrow(0, 0, top_k)
    out_slots = cfg.out_d.narrow(0, 0, top_k)
    out_row = out_slots[0].view(-1)[:width]  # routed sum lands in row 0
    gated = bool(getattr(mod, "gated", False))
    mg, mu, md = mod.multi_gate, mod.multi_up, mod.multi_down
    for i in range(bsz):
        if mod.gated:
            ext.exl3_mgemm(
                A[i], mg.ptrs_trellis, g_slots, mg.ptrs_suh, cfg.yh, mg.ptrs_svh,
                sel[i], None, mg.K, -1, mg.mcg, mg.mul1, mine, maxe, 0, 1, None, None)
        ext.exl3_mgemm(
            A[i], mu.ptrs_trellis, u_slots, mu.ptrs_suh, cfg.yh, mu.ptrs_svh,
            sel[i], None, mu.K, -1, mu.mcg, mu.mul1, mine, maxe, 0, 1, None, None)
        act_g = g_slots if mod.gated else u_slots
        mod.activation_fn_call(act_g, u_slots, a_slots, mod.act_limit)
        # A_had must not alias A (the autotuner relaunches on the first call);
        # the gate buffer is free after the activation
        ext.exl3_mgemm(
            a_slots, md.ptrs_trellis, out_slots, md.ptrs_suh, g_slots, md.ptrs_svh,
            sel[i], w[i], md.K, -1, md.mcg, md.mul1, mine, maxe, 0, 1, None, None)
        cfg.out_bszn[i].copy_(out_row)


def is_rocm() -> bool:
    try:
        import torch
        return getattr(torch.version, "hip", None) is not None
    except Exception:
        return False


def _moe_mt_native_arch(device) -> bool:
    """Cached-at-apply architecture envelope for the native RDNA GEMV path."""
    try:
        import torch
        arch = str(torch.cuda.get_device_properties(device).gcnArchName or "")
        arch = arch.split(":", 1)[0]
        # The ROCm RDNA path is compiled for gfx10/gfx11 families.  Querying
        # this only during apply keeps decode gating a dictionary lookup.
        return arch.startswith(("gfx10", "gfx11"))
    except Exception:
        return False


_applied = False
_applied_list: list[str] = []


def apply() -> list[str]:
    """Apply the ROCm patches. Idempotent; returns the list applied.

    Returns the same list on repeat calls rather than an empty one -- callers
    use this for reporting, and recomputing would make an already-patched
    process look unpatched.
    """
    global _applied, _applied_list
    if _applied:
        return _applied_list
    if not is_rocm() or not _env_on("EXL3_ROCM_PATCH", True):
        _applied = True
        return _applied_list
    _applied = True

    applied: list[str] = []


    # ------------------------------------------------------------------
    # arch_list: hipcc takes PYTORCH_ROCM_ARCH, not TORCH_CUDA_ARCH_LIST
    # ------------------------------------------------------------------
    # Setting TORCH_CUDA_ARCH_LIST on a ROCm build makes the JIT path pass
    # NVIDIA arch flags to hipcc. Harmless for a precompiled extension, wrong
    # for a source build.
    try:
        from ..util import arch_list as _al
        _orig_set_arch = _al.maybe_set_arch_list_env

        def _noop_arch_list(*args, **kwargs):
            return None

        _al.maybe_set_arch_list_env = _noop_arch_list
        applied.append("arch_list.maybe_set_arch_list_env -> no-op")
    except Exception as e:
        applied.append(f"!! FAILED arch_list: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Triton kernel binaries: "hsaco" on AMD, "cubin" on NVIDIA
    # ------------------------------------------------------------------
    # attention_fn/bc_attn.py's _compile_kernel does ck.asm["cubin"], which
    # KeyErrors on ROCm -- triton.backends.amd emits GPUTarget(backend='hip') and
    # names the code object "hsaco". The ext-side loader is already portable
    # (cuModuleLoadData maps to hipModuleLoadData, which takes an hsaco).
    #
    # Patched at triton.compile rather than at _compile_kernel because bc_mla.py
    # does `from .bc_attn import _compile_kernel` and so holds its own reference:
    # rebinding bc_attn's module attribute would fix one caller and miss the other.
    # bc_attn imports triton *inside* the function and calls triton.compile off the
    # module, so a single alias here covers every call site with no upstream code
    # duplicated.
    #
    # Aliasing rather than renaming: anything that legitimately wants "hsaco" still
    # finds it.
    try:
        import triton as _triton

        _orig_triton_compile = _triton.compile

        def _compile_alias_hsaco(*a, **kw):
            ck = _orig_triton_compile(*a, **kw)
            try:
                asm = ck.asm
                if "cubin" not in asm and "hsaco" in asm:
                    asm["cubin"] = asm["hsaco"]
            except Exception:
                pass
            return ck

        _triton.compile = _compile_alias_hsaco
        applied.append("triton.compile: alias asm['hsaco'] -> asm['cubin']")
    except ModuleNotFoundError:
        applied.append("triton not installed -- hsaco alias skipped")
    except Exception as e:
        applied.append(f"!! FAILED triton hsaco alias: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # DSA split kernel: RDNA tile/wave retune (spill + queue-stall fix)
    # ------------------------------------------------------------------
    # _dsa_attn_split_kernel at upstream's CUDA tuning (BLOCK_H=16, num_warps=4)
    # compiles on gfx1151 at the 256-VGPR ceiling with ~2050 VGPR spills and
    # 5332 B/work-item scratch -- the ONLY scratch user in the whole decode
    # stream. Every layer then alternates it against scratch-free kernels, and
    # the hardware queue pays a ~100-500us scratch-reconfiguration stall per
    # dispatch: ~41x/token = ~5 ms/token on DeepSeek-V4-Flash, measured
    # device-side (same stream, host parked in hipDeviceSynchronize, graphs-
    # immune -- see RDNA_NOTES "DS4 per-layer stall" and rocm_tools/
    # gap_profile.py). BLOCK_H=8 + num_warps=8 cuts spills to 438 and scratch
    # to 1756 B: the stalls collapse (1369 -> 45 big gaps / 31 tokens) and
    # decode goes 15.6 -> 17.8 t/s (+14%). Sweep of 16 variants in the notes;
    # H4/w16 and BLOCK_N=16 shapes spill less still but bench worse (14.9-17.0).
    #
    # BLOCK_H is a bc_dsa module constant, so it can be retuned here; the
    # split-kernel warps are an inline argument, so wrap _compile_kernel keyed
    # on the kernel NAME -- and rebind the wrapper in every module that did
    # `from .bc_attn import _compile_kernel` (bc_dsa, bc_mla), per the aliasing
    # note above. bc_mla's DSA-on-MLA path (GLM 5.2) hardcodes a local
    # BLOCK_H=16 inside _configure, so it gets only the warps half of the fix
    # (~1428 spills); untestable here regardless -- no model fits.
    # EXL3_ROCM_DSA_TUNE=0 restores upstream tuning.
    if _env_on("EXL3_ROCM_DSA_TUNE", True):
        try:
            from ..modules.attention_fn import bc_attn as _bca
            from ..modules.attention_fn import bc_dsa as _bcd
            from ..modules.attention_fn import bc_mla as _bcm

            _bcd.BLOCK_H = 8
            _orig_compile_kernel = _bca._compile_kernel

            def _compile_kernel_rdna(device, fn, signature, constexprs, num_warps, num_stages):
                if fn.__name__ == "_dsa_attn_split_kernel":
                    num_warps = 8
                return _orig_compile_kernel(device, fn, signature, constexprs, num_warps, num_stages)

            _bca._compile_kernel = _compile_kernel_rdna
            _bcd._compile_kernel = _compile_kernel_rdna
            _bcm._compile_kernel = _compile_kernel_rdna
            applied.append("DSA split kernel retuned for RDNA (BLOCK_H=8, num_warps=8; spills 2050->438)")
        except Exception as e:
            applied.append(f"!! FAILED DSA retune patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # MultiLinear (mgemm) fusion
    # ------------------------------------------------------------------
    # attn.py fuses K/V (and Q/G) into one MultiLinear, and mlp.py fuses
    # gate/up. Both dispatch to exl3_mgemm, whose cooperative grid is
    # dim3(num_sms, 1, concurrency) -- the shape that gets REFUSED outright when
    # it exceeds co-residency (verified: "too many blocks in cooperative
    # launch"). exl3_gemm is validated on this hardware; exl3_mgemm is not.
    #
    # RETIRED 2026-08-07 -- default is now OFF (i.e. mgemm ENABLED). Set
    # EXL3_ROCM_MGEMM=0 to restore the guard.
    #
    # The NaNs measured earlier the same day were not mgemm's. They were two
    # separate defects that have since been fixed:
    #   - hip_compat's __syncwarp mapped to a bare wave_barrier(), dropping the
    #     shared-memory ordering half of CUDA's contract
    #   - threadblock_reduce() in exl3_gemm_inner_rdna.hip.h read a different
    #     sh_c address than it wrote, off the end of the LDS block
    # With both fixed, GLM-4.6V generates coherent text through the mgemm paths,
    # while the guarded route degenerates into repetition. The guard is now the
    # thing producing bad output, so it is off by default.
    #
    # This is the second time this guard's stated reason turned out to be wrong
    # (it was inherited from the fork as "cooperative launch gets refused", then
    # re-justified as "kernel NaNs"). Retest before ever re-enabling it.
    if not _env_on("EXL3_ROCM_MGEMM", True):
        try:
            from ..modules import multilinear as _ml

            class _DisabledMultiLinear:
                """Sentinel that never constructs, so callers keep their None path."""
                def __new__(cls, *args, **kwargs):
                    return None

            from ..modules import attn as _attn
            from ..modules import mlp as _mlp
            _attn.MultiLinear = _DisabledMultiLinear
            _mlp.MultiLinear = _DisabledMultiLinear
            applied.append("MultiLinear fusion disabled (exl3_mgemm unvalidated on RDNA)")
        except Exception as e:
            applied.append(f"!! FAILED MultiLinear patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # bsz-1 MoE decode: off exl3_mgemm, onto the fused exl3_moe kernel
    # ------------------------------------------------------------------
    # The MultiLinear patch above only covers attn.py and mlp.py. BlockSparseMLP
    # builds its own MultiLinears and reaches exl3_mgemm by two further routes,
    # both of which fire at bsz == 1 -- i.e. every decode step of an MoE model:
    #
    #   block_sparse_mlp.py:1222  bszn_eligible -> self.bc.run_bszN(), whose
    #                             BC_BlockSparseMLP::run_bszN_gr is three
    #                             exl3_mgemm_gr calls (gate/up/down) captured
    #                             into a CUDA graph
    #   block_sparse_mlp.py:1319  the else fallback -- the same three calls,
    #                             ungraphed
    #
    # Observed on GLM-4.6V decode: the graph route dies with "Graph update
    # failed" (graph.cu:170) and then segfaults; the first sampled token is "!",
    # the argmax of a garbage logit row. Since mgemm is NaN on this hardware
    # (see above), fixing the graph bookkeeping would only buy a clean path to a
    # wrong answer, so both routes are closed rather than repaired.
    #
    # The escape is branch 1057, whose fourth clause is the only one a bsz == 1
    # call can satisfy: `not (support_quant_paths or bszn_eligible)`. That branch
    # runs ext.exl3_moe -- the fused kernel that prefills all 46 layers finite.
    # So clear exactly those two, and nothing else:
    #
    #   - is_quantized stays True. Forcing it False was the previous bug: it
    #     does not skip MoE, it reroutes to a dense path that cannot handle
    #     quantized weights and emits all-NaN.
    #   - Patch after load_local returns, so multi_gate/up/down and
    #     fused_mode_buffers are already built (all four are computed inside
    #     load_local, gated on support_quant_paths *at load time*).
    #     exl3_moe dereferences all of them.
    #
    # Keyed off EXL3_ROCM_MGEMM because it is the same kernel and the same
    # measurement; one switch should not lie about covering half the routes.
    #
    # RETIRED 2026-08-07 alongside the MultiLinear guard above, and for a sharper
    # reason: this reroute is now measurably WORSE than what it replaced. With the
    # guard on, GLM-4.6V decode degenerates into repetition; with it off (decode via
    # bc.run_bszN -> exl3_mgemm) the same model is coherent. At retirement the fused
    # exl3_moe path this patch forces was also numerically wrong; its two split-K
    # defects were fixed 2026-08-08 (see RDNA_NOTES.md, "exl3_gemm_inner_rdna.hip.h")
    # and it now matches an fp32 reference as closely as the per-expert path. The
    # reroute stays retired anyway: mgemm decode is correct and faster.
    if not _env_on("EXL3_ROCM_MGEMM", True):
        try:
            from ..modules import block_sparse_mlp as _bsq
            _bsq_cls = _bsq.BlockSparseMLP
            _orig_bsq_load = _bsq_cls.load_local

            def _load_no_mgemm_decode(self, *args, **kwargs):
                r = _orig_bsq_load(self, *args, **kwargs)
                self.support_quant_paths = False
                self.bc = None
                return r

            _bsq_cls.load_local = _load_no_mgemm_decode
            applied.append("bsz-1 MoE decode -> fused exl3_moe (exl3_mgemm NaNs on RDNA)")
        except Exception as e:
            applied.append(f"!! FAILED MoE bsz-1 patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # EXL3_ROCM_MOE_TORCH=1 -- bisect handle: MoE compute in pure torch
    # ------------------------------------------------------------------
    # Clearing fused_mode_buffers sets min_rows = 0 in the branch-1057 loop, so no
    # expert is skipped as "already claimed by the fused kernel" and every one falls
    # through to the Torch path -- self.ups[i].forward(), i.e. the per-expert exl3
    # Linear, which is the same GEMM/GEMV a dense model exercises correctly (verified
    # 2026-08-07: Gemma-4-31b generates coherent text on this build).
    #
    # Routing still runs ahead of this, so it isolates the MoE *compute* kernel alone:
    #   coherent -> exl3_moe is the fault
    #   garbage  -> the fault is upstream of it (routing, attention, RoPE, norms, or
    #               glm4v_moe architecture support), and MoE is exonerated
    #
    # Slow by construction (46 layers x top-8 experts of small matmuls per token).
    # Fine for a one-token prompt; not a mode to leave on.
    if _env_on("EXL3_ROCM_MOE_TORCH", False):
        try:
            from ..modules import block_sparse_mlp as _bst
            _bst_cls = _bst.BlockSparseMLP
            _orig_bst_load = _bst_cls.load_local

            def _load_torch_moe(self, *args, **kwargs):
                r = _orig_bst_load(self, *args, **kwargs)
                self.fused_mode_buffers = None
                return r

            _bst_cls.load_local = _load_torch_moe
            applied.append("MoE compute forced to torch path (EXL3_ROCM_MOE_TORCH bisect)")
        except Exception as e:
            applied.append(f"!! FAILED MoE torch patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # EXL3_ROCM_ROUTING_TORCH=1 -- bisect handle: expert routing in pure torch
    # ------------------------------------------------------------------
    # GLM-4.6V (and dots) set router_type="dots", so routing_dots runs
    # ext.routing_ds3_nogroup at *every* batch size -- a kernel built on
    # routing.cu's warp_radixsort_posf32_pl, which passes scores between lanes
    # through LDS. Dense models have no router at all, which is consistent with
    # Gemma-4-31b generating coherent text on this same build while GLM does not.
    #
    # routing_ds3 in the same module is a pure-torch implementation of the same
    # computation. GLM's config is n_group=1, topk_group=1, which collapses its
    # group mask to all-ones -- i.e. exactly the "nogroup" case the kernel
    # implements -- so it is a semantically equivalent drop-in, not an approximation.
    #
    #   coherent -> ext.routing_ds3_nogroup is the fault
    #   garbage  -> routing is exonerated and the fault is elsewhere in the
    #               glm4v_moe path (attention, RoPE, norms, architecture support)
    if _env_on("EXL3_ROCM_ROUTING_TORCH", False):
        try:
            from ..modules import block_sparse_mlp as _bsr
            _bsr_cls = _bsr.BlockSparseMLP
            _orig_bsr_load = _bsr_cls.load_local

            def _load_torch_routing(self, *args, **kwargs):
                r = _orig_bsr_load(self, *args, **kwargs)
                if getattr(self, "routing_fn", None) is _bsr.routing_dots:
                    self.routing_fn = _bsr.routing_ds3
                return r

            _bsr_cls.load_local = _load_torch_routing
            applied.append("expert routing forced to torch routing_ds3 (EXL3_ROCM_ROUTING_TORCH bisect)")
        except Exception as e:
            applied.append(f"!! FAILED routing torch patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # EXL3_ROCM_FORCE_TORCH=1 -- bisect handle: every exl3 Linear via reconstruct
    # ------------------------------------------------------------------
    # The current-tree equivalent of the fork's EXLLAMAV3_FORCE_TORCH_MODE. The fork
    # patched exl3.py directly (`bsz > 32 or FORCE_TORCH_MODE`); upstream restructured
    # that forward, so the same effect is achieved here by forcing params["reconstruct"].
    #
    # Upstream's default is rows <= AUTO_RECONSTRUCT_THRESHOLD (144) -> exl3 GEMM/GEMV
    # kernel, otherwise reconstruct + hgemm. Note the consequence: a short prompt runs
    # *prefill* through the quant kernels too, so "prefill is clean" was never evidence
    # that prefill used a different path from decode.
    #
    # Forcing it takes exl3_gemm and exl3_gemv out of the model entirely, leaving
    # dequant (reconstruct) + at::mm:
    #   coherent -> the fault is in exl3_gemm/exl3_gemv on this model's shapes
    #   garbage  -> those are exonerated; dequant, routing, attention or arch remain
    if _env_on("EXL3_ROCM_FORCE_TORCH", False):
        try:
            from ..modules.quant import exl3 as _x3
            _x3_cls = _x3.LinearEXL3
            _orig_x3_fwd = _x3_cls.forward

            def _forward_force_reconstruct(self, x, params, out_dtype = None):
                if not params.get("reconstruct"):
                    params = dict(params)
                    params["reconstruct"] = True
                return _orig_x3_fwd(self, x, params, out_dtype)

            _x3_cls.forward = _forward_force_reconstruct
            applied.append("all exl3 Linears forced through reconstruct+hgemm (EXL3_ROCM_FORCE_TORCH bisect)")
        except Exception as e:
            applied.append(f"!! FAILED force-torch patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Fused block-sparse MoE
    # ------------------------------------------------------------------
    # exl3_moe launches a NON-cooperative grid whose blocks must all be
    # co-resident for its group barriers, with nothing enforcing that. It has
    # also never been numerically validated. Forcing is_quantized False routes
    # MoE layers through the dense per-expert path.
    # Default OFF (i.e. fused MoE stays ENABLED). Measured 2026-08-07: forcing
    # is_quantized=False on EXL3-quantized tensors does not skip MoE, it reroutes
    # to a dense per-expert path that cannot handle quantized weights, and the
    # first MoE layer emits all-NaN. The fused kernel, by contrast, runs clean
    # through all 46 layers of GLM-4.6V. The fork carried this guard from an
    # older version; it is actively harmful here.
    if _env_on("EXL3_ROCM_MOE_DISABLE", False):
        try:
            from ..modules import block_sparse_mlp as _bs
            _cls = _bs.BlockSparseMLP
            # load_local is where is_quantized is computed (from the exl3 tensor
            # count), not load -- patching the wrong one silently does nothing.
            _orig_load = _cls.load_local

            def _load_no_fused_moe(self, *args, **kwargs):
                r = _orig_load(self, *args, **kwargs)
                self.is_quantized = False
                return r

            _cls.load_local = _load_no_fused_moe
            applied.append("fused block-sparse MoE disabled (exl3_moe unvalidated on RDNA)")
        except Exception as e:
            applied.append(f"!! FAILED MoE patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # v1.5.0: MoE decode for bsz <= MAX_BSZN -- the mgemm/GEMV route, restored
    # ------------------------------------------------------------------
    # Upstream v1.5.0 replaced BC_BlockSparseMLP.run_bszN's three exl3_mgemm
    # graph launches with exl3_moe_coop, a fused decode kernel built on
    # exl3_gemv_kernel.cuh (PTX mma + cp.async). It has no RDNA sibling; the
    # ROCm build links a stub that raises if reached (rocm/quant/
    # exl3_moe_coop_rdna.hip). block_sparse_mlp.forward takes that route
    # whenever `self.bc is not None and bsz <= MAX_BSZN`, and its else-branch
    # asserts bszn_eligible, so the dispatch has to be steered from outside.
    #
    # Two steers exist. The default restores upstream's own v1.4.4 route,
    # verbatim: per token, gate and up through exl3_mgemm (indices = that
    # token's experts), the activation, then down through exl3_mgemm with the
    # routing weights, which reduces the top-k expert outputs into out_d[0].
    # That row is the token's routed sum and is copied to out_bszn[i], where
    # the branch reads it back. Every call is num_tokens == 1, so on RDNA each
    # lands on the mgemv fast path (exl3_mgemv_rdna.hip: barrier-free dot
    # core, 120-240 GB/s). Measured 2026-09-22 on Laguna-S-2.1 4bpw
    # (bench_model.py): the fused steer decodes at 10.4 t/s, v1.4.4 on this
    # route at 20.8 -- the fused exl3_moe is a 16-row WMMA tile GEMM that pads
    # one useful row to sixteen and runs at a flat ~53 GB/s (RDNA_NOTES.md,
    # "Token generation was NOT memory-bound as first shipped").
    #
    # Mechanics: forward is wrapped so that, for the duration of the call,
    # self.bc is a proxy whose run_bszN is moe_mgemm_bszN (module level: the
    # v1.4.4 row loop, or the EXL3_ROCM_MOE_MULTI_TOKEN slot-major candidate
    # when it is on and supported); every other
    # attribute forwards to the real BC_BlockSparseMLP, so bszn_eligible still
    # sees a non-None bc and the per-expert graph paths (bsz > MAX_BSZN) still
    # reach the C++ object. Shared experts: upstream's kernel merges them in-
    # kernel and the forward tail skips them while self.bc_sh_exp is True, so
    # bc_sh_exp is forced False after load_local and the tail's Python
    # shared-expert path runs instead, as it does for every non-bszN tier.
    # Expert-range shards pass cfg.min_expert / max_expert exactly as
    # run_bszN does (num_tokens == 1 compacts out-of-range picks).
    #
    # EXL3_ROCM_MOE_MGEMM_ROUTE=0 selects the older steer: MAX_BSZN is zeroed
    # for the call and f_threshold set to 1, so bsz 1..8 runs the fused
    # exl3_moe kernel (correct per the 2026-08-08 fp32 comparison, half the
    # decode speed). EXL3_ROCM_MOE_BSZN=1 leaves upstream dispatch alone
    # (raises on ROCm) for the day exl3_moe_coop is ported.
    #
    # Raise only the Python per-token MoE proxy's row limit. Set it before
    # loading so scratch allocation and dispatch agree; shared MLP and native
    # BC limits remain unchanged. Twenty rows covers batch4 with four drafts.
    # Parse outside the fallback handler so invalid configuration fails early.
    moe_max_rows = _env_int(
        "EXL3_ROCM_MOE_MGEMM_MAX_ROWS",
        MOE_MGEMM_MAX_ROWS_DEFAULT, *MOE_MGEMM_MAX_ROWS_RANGE)
    if not _env_on("EXL3_ROCM_MOE_BSZN", False):
        try:
            from ..modules import block_sparse_mlp as _bsn
            from ..ext import exllamav3_ext as _ext
            _bsn_cls = _bsn.BlockSparseMLP
            _orig_bsn_load = _bsn_cls.load_local
            _orig_bsn_forward = _bsn_cls.forward
            _orig_bsn_unload = _bsn_cls.unload

            if _env_on("EXL3_ROCM_MOE_MGEMM_ROUTE", True):

                # Wire the module-level route (unchanged row loop plus the
                # experimental multi-token candidate) before any proxy call
                # can fire.
                import torch as _mt_torch
                from ..util.tensor import g_tensor_cache as _mt_tcache
                _MOE_MT["torch"] = _mt_torch
                _MOE_MT["ext"] = _ext
                _MOE_MT["tcache"] = _mt_tcache
                _MOE_MT["arch_ok"] = _moe_mt_native_arch
                _MOE_MT["arch_cache"] = {}
                _MOE_MT["max_rows"] = moe_max_rows

                if _env_on(MOE_MT_ENV, True):
                    # Resolve the architecture once here; the route gate reads
                    # a dict during decode and never queries GPU properties.
                    try:
                        _mt_any_native = False
                        if _mt_torch.cuda.is_available():
                            for i in range(_mt_torch.cuda.device_count()):
                                _MOE_MT["arch_cache"][i] = bool(
                                    _moe_mt_native_arch(f"cuda:{i}"))
                                _mt_any_native = _mt_any_native or _MOE_MT["arch_cache"][i]
                    except Exception:
                        _mt_any_native = False
                    if _mt_any_native:
                        _MOE_MT["on"] = True
                        _MOE_MT["reason"] = f"{MOE_MT_ENV} enabled on native gfx10/gfx11"
                        applied.append(
                            "MoE multi-token candidate ON (default; "
                            "EXL3_ROCM_MOE_MULTI_TOKEN=0 opts out: "
                            f"rows {MOE_MT_ROWS_MIN}..{moe_max_rows}, slots <= "
                            f"{MOE_MT_SLOTS_MAX} per row-aligned chunk, native gfx10/gfx11; "
                            "each loaded module owns "
                            "bounded [+128xHi half and small index/mask vectors] staging "
                            "and unload releases it (plus a bounded native bundle only "
                            "when existing cfg rows are short); sharded down "
                            "uses the native masked-reduce capability when available, "
                            "otherwise zeroes slot scratch and toggles "
                            f"{MOE_MT_FUSE_ENV}=0 around that call; R==1 and every "
                            "unsupported shape keep the unchanged row loop; "
                            "EXPERIMENTAL; validated for the measured "
                            "batch-1 TP2/MTP route by the r3 full/TP-half probe and "
                            "96 actual-model live-input cases (max candidate "
                            "relative-L2 3.9123e-4, scaled absolute error "
                            "6.6285e-4; all finite with env/output restoration), "
                            "with +8.055% observed / +11.236% engine throughput; "
                            "general batch shapes remain unvalidated; process/thread env toggle "
                            "is serialized only for candidate callers, arbitrary "
                            "concurrent native callers are unsupported, and TP2 rank "
                            "processes have separate environments -- gate: "
                            "rocm_tools/rdna2/moe_multitoken_probe.py)")
                    else:
                        applied.append(
                            f"!! {MOE_MT_ENV}=1 requested but no native gfx10/gfx11 device visible "
                            "-> multi-token candidate stays off (row loop unchanged)")
                else:
                    _MOE_MT["reason"] = f"{MOE_MT_ENV}=0 opt-out"

                class _BCProxy:
                    __slots__ = ("_bc", "_mod")

                    def __init__(self, bc, mod):
                        self._bc = bc
                        self._mod = mod

                    def __getattr__(self, name):
                        return getattr(self._bc, name)

                    def run_bszN(self, y, selected_experts, routing_weights):
                        moe_mgemm_bszN(self._mod, y, selected_experts, routing_weights)

                def _load_mgemm_route(self, *args, **kwargs):
                    r = _orig_bsn_load(self, *args, **kwargs)
                    self.bc_sh_exp = False
                    return r

                def _unload_mgemm_route(self, *args, **kwargs):
                    try:
                        return _orig_bsn_unload(self, *args, **kwargs)
                    finally:
                        moe_multi_token_scratch_clear(self)

                def _forward_mgemm_route(self, *args, **kwargs):
                    bc = self.bc
                    if bc is None:
                        return _orig_bsn_forward(self, *args, **kwargs)
                    self.bc = _BCProxy(bc, self)
                    try:
                        return _orig_bsn_forward(self, *args, **kwargs)
                    finally:
                        self.bc = bc

                _bsn_cls.load_local = _load_mgemm_route
                _bsn_cls.forward = _forward_mgemm_route
                _bsn_cls.unload = _unload_mgemm_route
                # The row cap, set ONCE here (patch time runs from `import
                # exllamav3`, before any module load) and only for this steer:
                # g_tensor_cache is exact-shape-keyed and never evicts, so the
                # cap must be final before the first load_local sizes scratch.
                # At the default 8 this is a no-op write.
                _bsn.MAX_BSZN = moe_max_rows
                applied.append(
                    f"MoE bsz<={moe_max_rows} decode -> per-token exl3_mgemm route "
                    "(v1.4.4's; mgemv fast path; exl3_moe_coop not ported)")
                if moe_max_rows != MOE_MGEMM_MAX_ROWS_DEFAULT:
                    applied.append(
                        f"MoE mgemm-route row cap {moe_max_rows} via "
                        "EXL3_ROCM_MOE_MGEMM_MAX_ROWS (Python block_sparse_mlp.MAX_BSZN "
                        "only; mlp.MAX_BSZN and the compiled native cap stay 8)")

            else:

                def _load_no_bszn(self, *args, **kwargs):
                    r = _orig_bsn_load(self, *args, **kwargs)
                    self.f_threshold = 1
                    return r

                def _forward_no_bszn(self, *args, **kwargs):
                    saved = _bsn.MAX_BSZN
                    _bsn.MAX_BSZN = 0
                    try:
                        return _orig_bsn_forward(self, *args, **kwargs)
                    finally:
                        _bsn.MAX_BSZN = saved

                _bsn_cls.load_local = _load_no_bszn
                _bsn_cls.forward = _forward_no_bszn
                applied.append("MoE bsz<=MAX_BSZN decode -> fused exl3_moe (EXL3_ROCM_MOE_MGEMM_ROUTE=0)")
                if moe_max_rows != MOE_MGEMM_MAX_ROWS_DEFAULT:
                    applied.append(
                        f"!! EXL3_ROCM_MOE_MGEMM_MAX_ROWS={moe_max_rows} ignored: the row cap "
                        "belongs to the mgemm route (inactive under EXL3_ROCM_MOE_MGEMM_ROUTE=0)")
        except Exception as e:
            applied.append(f"!! FAILED MoE bszN patch: {type(e).__name__}: {e}")
    elif moe_max_rows != MOE_MGEMM_MAX_ROWS_DEFAULT:
        applied.append(
            f"!! EXL3_ROCM_MOE_MGEMM_MAX_ROWS={moe_max_rows} ignored: native "
            "EXL3_ROCM_MOE_BSZN dispatch does not use the mgemm route")

    # ------------------------------------------------------------------
    # RDNA4 (gfx120x): fused MoE kernel unavailable -- per-expert fallback
    # ------------------------------------------------------------------
    # The fused exl3_moe kernel's WMMA sits on gfx11 intrinsics that have no
    # gfx12 encoding; rdna_wmma.hip.h compiles a __builtin_trap() there so the
    # comp units build, and this steer keeps the trap unreachable: clearing
    # fused_mode_buffers after load_local sets min_rows = 0 in the fused
    # branch. Multi-row BC expert projections also reach that WMMA wrapper,
    # so clear support_quant_paths for prefill and use the guarded Linear
    # reconstruct path. Keep bc: its ROCm per-token MGEMV proxy is valid
    # without cooperative WMMA and retains the fast decode route. Apply
    # this only to modules resident on the unsupported device.
    # EXL3_ROCM_RDNA4_FUSED_MOE=1 skips this steer (future gfx12 WMMA port).
    if not _env_on("EXL3_ROCM_RDNA4_FUSED_MOE", False):
        try:
            import torch as _t
            _is_gfx12 = _t.cuda.is_available() and any(
                _t.cuda.get_device_properties(i).gcnArchName.startswith("gfx12")
                for i in range(_t.cuda.device_count()))
            if _is_gfx12:
                from ..modules import block_sparse_mlp as _bs4
                _bs4_cls = _bs4.BlockSparseMLP
                _orig_bs4_load = _bs4_cls.load_local

                def _load_no_fused_gfx12(self, *args, **kwargs):
                    r = _orig_bs4_load(self, *args, **kwargs)
                    sample = next(iter(self.ups), None)
                    if sample is not None and not getattr(sample.inner, "cooperative_gemm_supported", True):
                        self.fused_mode_buffers = None
                        self.support_quant_paths = False
                    return r

                _bs4_cls.load_local = _load_no_fused_gfx12
                applied.append("RDNA4: fused MoE -> per-expert path (gfx11 WMMA has no gfx12 encoding)")
        except Exception as e:
            applied.append(f"!! FAILED RDNA4 MoE fallback: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # gfx1030 gated-MLP fp16 range balancing (silu(g)*u overflow)
    # ------------------------------------------------------------------
    # Qwen3-8B 4bpw on the V620 keeps gate/up finite (~314) but overflows
    # the fp16 product silu(g)*u (~9.9e4 > 65504) before down_proj: 3 of 8
    # reference cases go Inf/NaN (d-nonfinite-trace). The remedy rescales
    # the quant metadata of each eligible pair -- up svh /= 8, down svh *= 8
    # -- the same function algebraically (GEMM is linear in its input), with
    # 8x overflow headroom in the product; no clipping, no masking. It must
    # run after the deferred fills land (model_ls brackets each module's
    # load and writes checkpoint bytes into the svh buffers the freshly
    # built LinearEXL3 and the BC/MultiLinear pointer tables reference --
    # a load_local rescale is silently overwritten, verified). Hence the
    # lazy guard at forward entry, once per new inner (marker on the inner);
    # see mlp_range_balance.py for the full contract. The Model.load-
    # wrapping prototype measured full finiteness + 94.63% top-1 vs the
    # native BF16 oracle (d-quality-balanced8-loaded, 2026-09-28); GPU
    # verification of this guard is rocm_tools/rdna2/mlp_range_balance_probe.py.
    if _env_on("EXL3_ROCM_MLP_RANGE_BALANCE", True):
        try:
            from . import mlp_range_balance as _mrp
            note = _mrp.install()
            applied.append(f"gated-MLP svh range balance: {note}")
        except Exception as e:
            applied.append(f"!! FAILED gated-MLP range balance: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # v1.5.0: sliced Q/K/V bundle -- opt-in until validated
    # ------------------------------------------------------------------
    # attn.py, sliding_attn.py and gated_delta_net.py bundle every attention
    # projection into ONE exl3_mgemm launch of equal-width column slices
    # (SlicedMultiLinear; the C++ BC attention step takes the same tables).
    # The sliced mode -- per-source input Hadamard, strided B/C rows -- is
    # ported into exl3_gemm_kernel_rdna.hip.h / exl3_gemm_inner_rdna.hip.h but
    # has not been run on RDNA. Each module gates it on its own module-level
    # `_qkv_slice_enable` (upstream env EXL3_QKV_SLICE), read at load time, so
    # clearing that flag here leaves the v1.4.4 pairwise bundles in charge.
    # Numerically both routes compute the same projections.
    if not _env_on("EXL3_ROCM_QKV_SLICE", False):
        try:
            from ..modules import attn as _sa, sliding_attn as _ss, gated_delta_net as _sg
            n = 0
            for _m in (_sa, _ss, _sg):
                if hasattr(_m, "_qkv_slice_enable"):
                    _m._qkv_slice_enable = False
                    n += 1
            applied.append(f"sliced Q/K/V bundle (SlicedMultiLinear) off in {n} modules (mgemm sliced mode unvalidated on RDNA)")
        except Exception as e:
            applied.append(f"!! FAILED QKV slice patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # v1.5.0: batched expert reconstruct tier and MoE row-tile tiers -- opt-in
    # ------------------------------------------------------------------
    # BATCH_RECON (default on upstream) dequantizes heavy experts in batches
    # through reconstruct_had_batch / hgemm_batched (moe_batch_recon.py). Both
    # kernels are mechanical batch wrappers of code that runs here, but the
    # tier is unexercised on RDNA, so the v1.4.4 per-expert loop stays default.
    #
    # MTILE makes Python issue up to three fused-MoE launches per layer, one per
    # 16/32/64-row tier. exl3_moe_rdna.hip runs every tier through the 16-row
    # instance (the only one built), so the split only adds launches.
    # Both flags are module globals read at load / call time.
    try:
        from ..modules import block_sparse_mlp as _bst2
        batch_choice = os.environ.get("EXL3_ROCM_BATCH_RECON", "0")
        if batch_choice == "auto":
            _bst2.BATCH_RECON = os.environ.get("EXL3_MOE_BATCH_RECON", "1") != "0"
            _bst2.ROCM_BATCH_RECON_AUTO = True
            applied.append("batched expert reconstruction auto: exact validated gfx1030 TP2 geometry only")
        elif not _env_on("EXL3_ROCM_BATCH_RECON", False):
            _bst2.BATCH_RECON = False
            applied.append("batched expert reconstruction off (EXL3_ROCM_BATCH_RECON=auto for validated geometry)")
        if not _env_on("EXL3_ROCM_MOE_MTILE", False):
            _bst2.MTILE = False
            applied.append("fused-MoE row-tile tiers off (only the 16-row instance exists on RDNA)")
    except Exception as e:
        applied.append(f"!! FAILED batch-recon/mtile patch: {type(e).__name__}: {e}")

    globals()['_applied_list'] = applied
    return applied


def describe() -> str:
    return "\n".join(f"  - {p}" for p in apply()) or "  (no ROCm patches active)"
