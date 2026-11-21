"""gfx1030 gated-MLP fp16 range balance (rocm_py patch).

The problem (d-reference, 2026-09-28): EXL3 Qwen3-8B 4bpw on the V620 keeps
its gate/up projections finite (~314), but the fp16 product silu(g)*u
(~9.9e4) overflows before down_proj -- 24 Inf values at model.layers.2.mlp;
3 of 8 reference cases go Inf/NaN. This is intermediate dynamic range, not
kernel arithmetic: clipping or non-finite masking would change the function
and is refused.

The fix, per eligible (up, down) pair, applied once after the weights have
actually landed:

    up.inner.svh   /= 8    -> u, and hence a = silu(g)*u, sits 8x lower
    down.inner.svh *= 8    -> down's output scale cancels it again

svh is EXL3's output-side scale vector (folded into reconstruction / the
final had_r_128; the bias is added after it, hence bias-free pairs only) and
the GEMM is linear in its input, so a/8 through an 8x output scale is the
same function up to fp16 rounding. The gate is untouched: the product gains
8x headroom, not 64x (only u is divided). No checkpoint rewrite, no
per-token math kernels.

Why the guard sits at forward entry: model_ls brackets every module load
with stc.begin/end_deferred_load, and the deferred fill rewrites the very
svh buffers load_local handed to the BC/MultiLinear pointer tables -- a
rescale there is silently overwritten (verified; that is why the prototype
scaled after Model.load returned). The first forward of a module is
guaranteed post-fill and also covers load_gen, per-slice reload and
pin/unpin. The original forward -- BC and MultiLinear dispatch included --
does all the math; this only adjusts the metadata it reads.

Contract:
  * Once per new LinearEXL3 inner: the marker is an attribute on the inner
    object. load_exl3 rebuilds the inner per load, so a reload re-balances
    exactly once from fresh data; pin_linears keeps inner and its (moved,
    not refilled) svh, so it does not. No global registry -- nothing here
    retains weight objects after unload (the probe weakrefs inners to keep
    that honest). One side marked without the other is an unresolvable
    history: raise.
  * In-place div_/mul_ under torch.inference_mode() only: BC_LinearEXL3,
    MultiLinear and CUDA graphs read svh by raw storage pointer.
  * Validate before touching anything (fp16, contiguous, finite, headroom
    on the multiply side); on failure raise with the weights untouched.
  * Scope: HIP build + gfx1030 input device; EXL3 up/down; act_limit == 0
    (a swiglu clamp on u would move its threshold under /8); no up/down
    bias (added after the svh stage); dense GatedMLP only (BlockSparseMLP/
    MoE untouched). params["ovr"] (LoRA-style overrides) and LoRA deltas
    are conflicts: a partial override would mix scaled base data with
    unscaled override data. Conflict + unbalanced pair -> decline to
    balance; conflict + already-balanced pair -> raise, since continuing
    would silently corrupt the unsupported configuration.
  * Scale fixed at 8.0 (the proven factor); A/B is the whole-patch switch
    EXL3_ROCM_MLP_RANGE_BALANCE=0, read at import in rocm_py.apply().

Provenance: the prototype (Model.load wrapper; /work/range_balance_after_load.py)
produced the first full-finite 1024-position evaluation: top-1 969/1024 =
94.63% against the native BF16 oracle (d-quality-balanced8-loaded.json).
That measurement belongs to the prototype; this lazy forward-entry guard
re-expresses it under the rocm_py convention and its GPU verification is
rocm_tools/rdna2/mlp_range_balance_probe.py.
"""

from __future__ import annotations

# The reciprocal power-of-two pair. Fixed at the proven factor; changing it
# means re-collecting the oracle comparison.
SCALE = 8.0

# Attribute set on a LinearEXL3 inner once its svh has been rescaled; the
# value is the scale applied (introspection: probes, debugging).
MARKER = "_rocm_py_mlp_range_balance_scale"

# Attribute set on the wrapped forward so install()/apply() are idempotent.
WRAPPER = "_rocm_py_mlp_range_balance_wrapper"

_ARCH_PREFIX = "gfx1030"

_gfx1030_by_index: dict[int, bool] = {}
_is_hip: bool | None = None


def arch_supported(gcn_arch_name: str) -> bool:
    """gfx1030 (the V620 this port targets) only. gfx1031/32/34/35 share the
    family but have no fp16-range measurement behind them; the balance is a
    numerics policy tied to a measured failure, so start conservative."""
    return str(gcn_arch_name).startswith(_ARCH_PREFIX)


def device_is_gfx1030(device) -> bool:
    """Cached per-device gcnArchName probe; anything this cannot read is
    treated as 'not gfx1030' (skip)."""
    import torch
    if isinstance(device, str):
        device = torch.device(device)
    if getattr(device, "type", None) != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    ok = _gfx1030_by_index.get(index)
    if ok is None:
        try:
            props = torch.cuda.get_device_properties(index)
            ok = arch_supported(getattr(props, "gcnArchName", ""))
        except Exception:
            ok = False
        _gfx1030_by_index[index] = ok
    return ok


def is_hip_build() -> bool:
    global _is_hip
    if _is_hip is None:
        try:
            import torch
            _is_hip = getattr(torch.version, "hip", None) is not None
        except Exception:
            _is_hip = False
    return _is_hip


def override_state(params, up, down) -> str:
    """
    Coverage of one (up, down) pair by the params["ovr"] override table
    (LoRA-style hot-swap; LinearEXL3.forward routes to the override inner
    when ovr[key].inner is not self):

        "none"    base linears are what will run
        "full"    both sides override with their own inners: base svh is
                  inert while the override is active, balancing it is safe
                  for the later un-overridden forwards
        "partial" exactly one side overrides: scaled base data would mix
                  with unscaled override data (or vice versa)
    """
    ovr = params.get("ovr") if isinstance(params, dict) else None
    if not ovr:
        return "none"
    sides = []
    for lin in (up, down):
        o = ovr.get(getattr(lin, "key", None))
        sides.append(o is not None and getattr(o, "inner", None) is not lin.inner)
    if sides[0] and sides[1]:
        return "full"
    if sides[0] != sides[1]:
        return "partial"
    return "none"


def dynamic_conflict(module, up, down, params) -> str | None:
    """Reason this pair must not be balanced because of per-call state
    (LoRA deltas added after svh; partial override), or None."""
    if getattr(up, "lora_a_tensors", None) or getattr(down, "lora_a_tensors", None):
        return "a LoRA delta is added to the pair after the svh stage"
    if override_state(params, up, down) == "partial":
        return f"a params[\"ovr\"] override covers only one side of {getattr(module, 'key', '?')}"
    return None


def classify_pair(module, up, down, params = None) -> tuple[str, str]:
    """
    Range-balance status of one (up, down) GatedMLP slice pair:

        "already"   both inners marked -> nothing to do (fast path)
        "mixed"     one side marked -> inconsistent history; caller raises
        "skip"      out of scope: unloaded / non-exl3 / bias / act_limit
        "conflict"  in scope but unsafe configuration right now (lora/ovr)
        "eligible"  standard gfx1030 EXL3 pair, unbalanced; safe to balance

    Device scope is NOT checked here; the forward guard checks the input
    tensor's device once per call.
    """
    ui = getattr(up, "inner", None)
    di = getattr(down, "inner", None)
    if ui is None or di is None:
        return "skip", "unloaded"
    u_marked = getattr(ui, MARKER, None) is not None
    d_marked = getattr(di, MARKER, None) is not None
    if u_marked and d_marked:
        return "already", "balanced after the deferred fills landed"
    if u_marked != d_marked:
        side = "up" if u_marked else "down"
        return "mixed", f"{side}.inner carries a range-balance marker the other lacks"
    if getattr(module, "act_limit", 0.0):
        return "skip", "act_limit clamps u; /8 would move the threshold"
    if getattr(up, "quant_type", None) != "exl3" or getattr(down, "quant_type", None) != "exl3":
        return "skip", "not an EXL3 pair"
    if ui.bias is not None or di.bias is not None:
        return "skip", "bias is added after the svh stage and would not rescale"
    conflict = dynamic_conflict(module, up, down, params)
    if conflict is not None:
        return "conflict", conflict
    return "eligible", ""


def balance_pair(up, down, scale: float = SCALE) -> str:
    """
    Validate, then rescale one pair's output-scale metadata in place:
    up.svh /= scale, down.svh *= scale, marking both inner objects.
    Validation failures raise BEFORE anything is mutated. Not idempotent at
    this level by design -- callers gate with classify_pair; double-calling
    compounds, so only the once-per-inner marker should drive it.
    """
    import torch
    ui, di = up.inner, down.inner
    su, sd = ui.svh, di.svh
    ukey, dkey = getattr(up, "key", "?"), getattr(down, "key", "?")
    with torch.inference_mode():
        if su.dtype != torch.half or sd.dtype != torch.half:
            raise RuntimeError(
                f"mlp range balance {ukey}/{dkey}: svh dtype "
                f"{su.dtype}/{sd.dtype} != torch.half; refusing to touch "
                f"metadata this patch was not measured against")
        if not (su.is_contiguous() and sd.is_contiguous()):
            raise RuntimeError(
                f"mlp range balance {ukey}/{dkey}: svh non-contiguous; the "
                f"in-place rescale and the pointer-table kernels assume "
                f"contiguous storage; refusing")
        fp16_max = float(torch.finfo(torch.half).max)
        max_u = torch.max(su.abs())
        max_d = torch.max(sd.abs())
        if not (bool(torch.isfinite(max_u)) and bool(torch.isfinite(max_d))):
            raise RuntimeError(
                f"mlp range balance {ukey}/{dkey}: non-finite svh before "
                f"balancing ({float(max_u)} / {float(max_d)}); the checkpoint "
                f"is corrupt, rescaling cannot fix that")
        if float(max_d) * scale > fp16_max:
            raise RuntimeError(
                f"mlp range balance {ukey}/{dkey}: no fp16 headroom, down "
                f"svh absmax {float(max_d)} * {scale} > {fp16_max}; "
                f"weights left untouched")
        scaled_up = su / scale
        if bool(((su != 0) & (scaled_up == 0)).any()):
            raise RuntimeError(
                f"mlp range balance {ukey}/{dkey}: up svh would underflow "
                f"to zero; weights left untouched")
        su.copy_(scaled_up)
        sd.mul_(scale)
        setattr(ui, MARKER, scale)
        setattr(di, MARKER, scale)
        # Pre-checks make this near-impossible; it exists because silently
        # shipping inf scales is worse. Marked first so a retry cannot
        # compound; the message tells the operator to reload.
        if not (bool(torch.isfinite(su).all()) and bool(torch.isfinite(sd).all())):
            raise RuntimeError(
                f"mlp range balance {ukey}/{dkey}: svh became non-finite "
                f"during rescaling; the pair is marked (no compounding) but "
                f"the model must be reloaded before it is trusted")
    return f"{ukey}/{dkey}: svh /= {scale:g}, *= {scale:g}"


def balance_module(module, params = None) -> None:
    """
    Balance every eligible pair of one GatedMLP. Device-agnostic core of
    the forward guard, kept separate so CPU tests exercise the real
    bookkeeping without pretending to own a V620.

    A pair that is already balanced but now faces an unsupported
    configuration (LoRA delta / partial override) raises: its scaled
    metadata would silently corrupt that configuration, and there is no
    safe way to un-apply it mid-run -- reload resets it.
    """
    for up, down in zip(getattr(module, "ups", None) or [],
                        getattr(module, "downs", None) or []):
        status, note = classify_pair(module, up, down, params)
        if status == "eligible":
            balance_pair(up, down)
        elif status == "mixed":
            raise RuntimeError(
                f"mlp range balance {getattr(module, 'key', '?')}: {note}; "
                f"one side would be rescaled against the other's history. "
                f"Unload and reload the model to reset both.")
        elif status == "already":
            conflict = dynamic_conflict(module, up, down, params)
            if conflict is not None:
                raise RuntimeError(
                    f"mlp range balance {getattr(module, 'key', '?')}: the "
                    f"pair is already balanced but {conflict}; continuing "
                    f"would apply the rescale to an unsupported "
                    f"configuration. Reload the model to reset the pair.")


def _maybe_balance(module, x, params = None) -> None:
    """Forward-entry guard: HIP + gfx1030 input device, then balance."""
    if not is_hip_build():
        return
    if not device_is_gfx1030(x.device):
        return
    balance_module(module, params)


def wrap_forward(cls) -> str:
    """Wrap ``cls.forward`` with the guard; the original still performs all
    math and BC/MultiLinear dispatch. Idempotent: a wrapped forward carries
    the WRAPPER flag and is never double-wrapped."""
    orig = cls.forward
    if getattr(orig, WRAPPER, False):
        return f"{cls.__name__}.forward already guarded"
    import functools

    @functools.wraps(orig)
    def _forward_balanced(self, x, params, out_dtype = None):
        _maybe_balance(self, x, params)
        return orig(self, x, params, out_dtype)

    setattr(_forward_balanced, WRAPPER, True)
    cls.forward = _forward_balanced
    return f"{cls.__name__}.forward guarded (scale {SCALE:g})"


def install() -> str:
    """Apply the patch (called from rocm_py.apply() under the env switch)."""
    from ..modules.mlp import GatedMLP
    return wrap_forward(GatedMLP)


def audit(model) -> dict:
    """
    Walk a loaded model and classify every GatedMLP pair without touching
    anything -- the probe's introspection hook. Returns per-status counts
    and the eligible-but-unbalanced keys.
    """
    from ..modules.mlp import GatedMLP
    counts = {"gated_mlp": 0, "pairs": 0, "already": 0, "eligible": 0,
              "skip": 0, "mixed": 0, "conflict": 0}
    unbalanced: list[str] = []
    scales: set[float] = set()
    for m in model:
        if not isinstance(m, GatedMLP):
            continue
        counts["gated_mlp"] += 1
        for up, down in zip(m.ups, m.downs):
            counts["pairs"] += 1
            status, _ = classify_pair(m, up, down)
            counts[status] += 1
            if status == "eligible":
                unbalanced.append(f"{getattr(up, 'key', '?')}|{getattr(down, 'key', '?')}")
            elif status == "already":
                scales.add(float(getattr(up.inner, MARKER)))
    return {"counts": counts, "unbalanced_eligible": unbalanced,
            "marked_scales": sorted(scales)}
