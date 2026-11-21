from __future__ import annotations
import torch
from ...model.config import Config
from .exl3_lib.quantize import preapply_had_l, preapply_had_r, had_k, had_n, codebook_mul1_mult
from ...ext import exllamav3_ext as ext
from ...util.tensor import g_tensor_cache
import os
from ...util import profile_opt

# Fused Triton INT8 helpers (C2: kill the torch-composed quant/epilogue
# passes). Optional: when unavailable the torch reference path below runs.
try:
    from ...rocm_py.dense_int8 import dense_epilog_had, dquant_rowwise, quant_w8_slab
    _dense_triton = True
except Exception:
    _dense_triton = False

AUTO_RECONSTRUCT_THRESHOLD = 144
MAX_RECONSTRUCT_SLICE_N = 32768
RECONSTRUCT_SLICE_GRANULARITY_N = 128

no_fused_reconstruct = os.environ.get("EXL3_NO_FUSED_RECONSTRUCT", "0") != "0"

# INT8 W8A8 prefill path (experimental, default off). When enabled, eligible
# linears run the rotated basis (input Hadamard, same as the non-fused path)
# with the inner trellis weights rounded once to a per-output-channel int8
# grid (scales folded into a cached svh copy, so the epilogue is unchanged)
# and activations quantized to int8 per token. Accumulation is exact int32
# (K <= tens of thousands cannot overflow: K * 127 * 127 < 2^31).
# The math is backend-agnostic torch ops plus one int8->int32 GEMM; on ROCm
# that GEMM is ext.igemm_recon (rocBLAS int8), elsewhere an exact emulation.
int8_prefill = os.environ.get("EXL3_INT8_PREFILL", "0") != "0"
# Fixed global int8 grid for mul1 models without a stored s0: the mul1 value is
# (887*B - 452608) * 2^-17 with B in 0..1020, so a step of 4 * 887 * 2^-17
# maps B onto [-128, 127] with no per-tensor/per-channel scale. Experimental.
_int8_fixed_grid = os.environ.get("EXL3_INT8_FIXED_GRID", "0") != "0"
INT8_FIXED_S0 = 4.0 * 887.0 / 131072.0
# Weight-only emulation (evaluation): while EMU_W["on"] is set (the MoE emulation
# path turns it on around the expert MLPs), every mul1 linear decodes its weights,
# rounds them to the fixed grid (unless EXL3_INT8_EMU_NOROUND=1, the control) and runs
# an fp32 matmul in the rotated basis. Speed is irrelevant here.
EMU_W = {"on": False}
_emu_noround = os.environ.get("EXL3_INT8_EMU_NOROUND", "0") != "0"

# K-block size for the exact-int32 emulation GEMM: partial sums stay below
# 2^24, so every fp32 block product is bit-exact and the fp64 tail is too.
_INT8_EMU_KBLOCK = 512

# Ablation for error attribution (dumps only, never for speed):
# "" = full W8A8, "w" = int8 weights + fp32 activations, "a" = fp16 weights +
# int8 activations. The single-side paths use fp32 matmuls on dequantized
# operands, so each measures exactly one grid's error.
_int8_ablate = os.environ.get("EXL3_INT8_ABLATE", "")
# Minimum row count for INT8 engagement (perf heuristic; accuracy work such
# as KLD can lower it to bring more cases onto the int8 path).
_int8_min_rows = int(os.environ.get("EXL3_INT8_MIN_ROWS", "1024"))
# Evaluation scope: "" = all eligible linears, "head" = only vocab-sized N
# (lm_head), "nohead" = everything except the lm_head.
_int8_scope = os.environ.get("EXL3_INT8_SCOPE", "")


def _hgemm_fp16_out() -> bool:
    # Fair-fp16 baseline switch (read per call so a same-proc A/B can flip
    # it between forwards). When set, the fp16 reconstruct path accumulates
    # the GEMM in fp16 and upcasts to the requested (fp32) contract dtype
    # afterwards, instead of running hgemm with fp32 output. The residual
    # stream contract is unchanged (callers still receive fp32); only the
    # GEMM arithmetic changes. The INT8 path is untouched, so INT8 runs
    # bit-identically with the switch on or off. Default off = legacy
    # fp32-accumulation behaviour. Decode (bc.run_alloc) is unaffected.
    return os.environ.get("EXL3_HGEMM_FP16_OUT", "0") != "0"

# Shapes that already logged int8 engagement (one line per unique GEMM shape).
_int8_logged_shapes = set()


def _exact_int32_gemm(a8: torch.Tensor, w8: torch.Tensor) -> torch.Tensor:
    # Bit-exact (a8 int8) @ (w8 int8) in int32, via K-blocked fp32 products
    # accumulated in fp64. Exact: products fit 15 bits, block sums fit 23.
    m, k = a8.shape
    n = w8.shape[1]
    acc = torch.zeros((m, n), dtype=torch.float64, device=a8.device)
    af = a8.float()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for k0 in range(0, k, _INT8_EMU_KBLOCK):
            acc += af[:, k0:k0 + _INT8_EMU_KBLOCK] @ w8[k0:k0 + _INT8_EMU_KBLOCK, :].float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return acc.to(torch.int32)


class LinearEXL3:

    quant_type: str = "exl3"

    @property
    def cooperative_gemm_supported(self):
        # The gfx12 WMMA wrapper intentionally traps. It is reached by the
        # small multi-row GEMM too, including Qwen's vision merger. Cache this
        # per weight/device; single-row decode and gfx10/gfx11 remain native.
        cache = getattr(self, "_cooperative_gemm_support", None)
        if cache is None:
            cache = self._cooperative_gemm_support = {}
        device = self.trellis.device
        if device not in cache:
            supported = True
            if torch.version.hip and device.type == "cuda":
                arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
                supported = not arch.startswith("gfx12")
            cache[device] = supported
        return cache[device]

    def __init__(
        self,
        config: Config | None,
        in_features: int,
        out_features: int,
        scale: torch.Tensor | None = None,
        su: torch.Tensor | None = None,
        sv: torch.Tensor | None = None,
        suh: torch.Tensor | None = None,
        svh: torch.Tensor | None = None,
        trellis: torch.Tensor | None = None,
        mcg: torch.Tensor | None = None,
        mul1: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        transformers_fix: bool = False,
        key: str | None = None,
        s0: torch.Tensor | None = None
    ):
        assert scale is None, "scale is no longer used"
        assert su is not None or suh is not None, "either su (packed) or suh (unpacked) is required"
        assert sv is not None or svh is not None, "either sv (packed) or svh (unpacked) is required"
        assert trellis is not None, "trellis is required"
        if su is not None: assert su.dtype == torch.int16, "su is wrong datatype"
        if sv is not None: assert sv.dtype == torch.int16, "sv is wrong datatype"
        if suh is not None: assert suh.dtype == torch.half, "suh is wrong datatype"
        if svh is not None: assert svh.dtype == torch.half, "svh is wrong datatype"
        assert trellis.dtype == torch.int16, "trellis is wrong datatype"
        assert len(trellis.shape) == 3, "trellis must have dim = 3"

        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)

        # Not a Module subclass, so the config-or-NullConfig default doesn't apply here; TP imports pass
        # config=None and forward() reads config.infer_params
        if config is None:
            from ...model.config import NullConfig
            config = NullConfig()
        self.config = config
        self.transformers_fix = transformers_fix
        self.key = key

        # self.scale = scale.item()
        self.su = None
        self.sv = None
        self.suh = suh if suh is not None else self.unpack_bf(su)
        self.svh = svh if svh is not None else self.unpack_bf(sv)
        self.trellis = trellis
        self.K = trellis.shape[-1] // 16
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.out_dtype = out_dtype
        self.default_out_dtype = out_dtype or torch.half

        self.mcg_tensor = mcg
        self.mul1_tensor = mul1
        self.mcg = self.mcg_tensor is not None
        self.mul1 = self.mul1_tensor is not None
        # The multiplier (and with it the tile order) is a build-time property of the ext: a model made for the other
        # format would decode to noise.
        if self.mul1:
            assert (int(self.mul1_tensor.item()) & 0xFFFFFFFF) == codebook_mul1_mult, \
                f"{key}: model multiplier {int(self.mul1_tensor.item()) & 0xFFFFFFFF:#x} != ext build {codebook_mul1_mult:#x}"

        # INT8-exact codebook grid step (round 2 of the W8A8 prefill work).
        # Present only in models encoded with the int8 grid; the decoded tile
        # values are then exactly q * s0, so the runtime reconstructs int8
        # directly and the stock decode path rounds to the grid. Old models
        # (s0 None) behave exactly as before.
        if s0 is not None:
            assert s0.numel() == 1, "s0 must be a scalar grid step"
            s0 = s0.to(dtype = torch.float32).flatten()
            assert s0.item() > 0.0, "s0 must be positive"
            self.s0_value = float(s0.item())
        else:
            self.s0_value = 0.0
        if s0 is None and _int8_fixed_grid and self.mul1:
            s0 = torch.tensor([INT8_FIXED_S0], dtype = torch.float32)
            self.s0_value = INT8_FIXED_S0
        self.s0 = s0

        self._fused_reconstruct = None
        self.bsz1_xh_args = (self.trellis.device, (1, self.in_features), self.out_dtype)
        self.bc = ext.BC_LinearEXL3(
            self.trellis,
            self.suh,
            self.svh,
            self.K,
            self.bias,
            self.mcg,
            self.mul1,
            g_tensor_cache.get(*self.bsz1_xh_args)
        )


    def unload(self):
        # g_tensor_cache.drop(*self.bsz1_xh_args)
        pass


    def get_tensors(self, key: str):
        return {
            f"{key}.{subkey}": tensor.contiguous()
            for subkey, tensor in [
                ("su", self.su),
                ("sv", self.sv),
                ("suh", self.suh),
                ("svh", self.svh),
                ("trellis", self.trellis),
                ("bias", self.bias),
                ("mcg", self.mcg_tensor),
                ("mul1", self.mul1_tensor),
                ("s0", self.s0),
            ] if tensor is not None
        }


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if "ovr" in params:
            ovr = params["ovr"]
            if self.key in ovr and ovr[self.key].inner is not self:
                return ovr[self.key].forward(x, params, out_dtype)

        if EMU_W["on"] and self.mul1:
            return self._emu_w(x, out_dtype)

        # The EXL3 kernels read x as contiguous rows; a strided view (e.g. a head-group slice
        # of a wider tensor) would be silently misread as interleaved garbage. Producers are
        # responsible for contiguity (a silent copy here would hide a hot-path inefficiency
        # and break CUDA-graph address stability)
        assert x.is_contiguous(), f"LinearEXL3 {self.key}: non-contiguous input {tuple(x.shape)}"

        reconstruct = params.get("reconstruct")
        if not reconstruct:
            rows = x.numel() // x.shape[-1]
            if rows <= AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
                if rows > 1 and not self.cooperative_gemm_supported:
                    return self.reconstruct_hgemm(x, out_dtype, params)
                dtype = out_dtype or self.default_out_dtype
                return self.bc.run_alloc(x, self.out_features, dtype == torch.float)

        return self.reconstruct_hgemm(x, out_dtype, params)


    def unpack_bf(self, bitfield: torch.Tensor):
        # For some reason this operation causes a GPU assert on Transformers. Running on CPU seems to fix it
        device = bitfield.device
        if self.transformers_fix:
            bitfield = bitfield.cpu()

        # (Only used for full reconstruct and loading old models, not during inference)
        bitfield = bitfield.view(torch.uint16).to(torch.int)
        masks = (1 << torch.arange(16)).to(bitfield.device)
        expanded = (bitfield.unsqueeze(-1) & masks) > 0
        expanded = expanded.flatten()
        # NOT torch.where with CPU scalar tensors: that path misses the device guard when the
        # condition lives on a non-current device (observed on torch 2.11.0+cu130) — the kernel
        # launches on the current device's context, faults there, silently zero-fills the output
        # and leaves every other device in the process unusable. Map bool -> {-1, +1} arithmetically
        expanded = 1.0 - expanded.to(torch.float16) * 2.0
        return expanded.contiguous().to(device)


    def reconstruct_hgemm(self, x: torch.Tensor, out_dtype, params: dict | None = None):

        shape = x.shape
        rows = x.numel() // shape[-1]
        out_shape = shape[:-1] + (self.out_features,)
        x = x.view(rows, self.in_features)
        y = torch.empty(out_shape, dtype = out_dtype or self.default_out_dtype, device = x.device)

        y_ = y.view(rows, self.out_features)

        # INT8 W8A8 path: rotated basis with int8 inner weights and
        # per-token int8 activations. Engaged at ALL row counts when the flag
        # is on (the old rows<1024-only gate is removed); the fp16 fused
        # original-basis path below stays the default when the flag is off.
        # Returns None when the int8 route declines (ablation + sliced N),
        # in which case execution falls through to the fp16 path.
        if (int8_prefill and self._int8_eligible()
                and (params is None or "capture" not in params)):
            r = self._reconstruct_hgemm_int8(x, y, y_, rows, out_dtype)
            if r is not None:
                return r

        # Fair-fp16 baseline (EXL3_HGEMM_FP16_OUT=1): run the fp16 GEMM
        # body into an fp16 buffer, then upcast to the requested (fp32)
        # contract dtype. The residual stream still receives fp32; the only
        # change is GEMM accumulation/output precision. The INT8 path above
        # already returned, so it is unaffected by this switch.
        req = out_dtype or self.default_out_dtype
        if req == torch.float and _hgemm_fp16_out():
            yh = torch.empty(out_shape, dtype = torch.half, device = x.device)
            self._reconstruct_hgemm_fp16(x, yh, yh.view(rows, self.out_features), rows)
            with torch.inference_mode():
                y_.copy_(yh.view(rows, self.out_features))
            del yh
        else:
            self._reconstruct_hgemm_fp16(x, y, y_, rows)

        if self.bias is not None:
            y += self.bias
        return y


    def _reconstruct_hgemm_fp16(self, x, y, y_, rows):
        # fp16 GEMM body shared by the legacy fp32-out path and the
        # EXL3_HGEMM_FP16_OUT baseline. Writes the GEMM result (plus the
        # output Hadamard for the non-fused path) into y_/y; the caller owns
        # the buffer dtype and any upcast. No bias handling here.
        del y
        # Fused path: reconstruct emits ORIGINAL-basis weights (both Hadamards + sign
        # vectors folded into the memory-bound reconstruct kernel), so the gemm runs on the
        # raw input and the standalone input/output had_r_128 launches disappear (~14% of
        # long-chunk prefill GPU time). Requires 128-divisible dims (always true for EXL3
        # tensors: both sides are had-transformed at quant time)
        if self._fused_reconstruct is None:
            self._fused_reconstruct = (
                self.in_features % 128 == 0 and self.out_features % 128 == 0
                and not no_fused_reconstruct
            )

        # The fused kernel costs ~4x plain reconstruct (k*n-proportional) while the saved
        # had launches scale with rows*(k+n); breakeven is rows ~400-900 across shapes
        use_fused = self._fused_reconstruct and rows >= 1024

        if use_fused:
            xh = x
        else:
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, self.suh, None, 1.0)

        if self.out_features <= MAX_RECONSTRUCT_SLICE_N:
            w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
            if use_fused:
                ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh, self.K, self.mcg, self.mul1, 0)
            else:
                self._recon(w)
            ext.hgemm_recon(xh, w, y_)
        else:
            numel_ = self.in_features * MAX_RECONSTRUCT_SLICE_N
            w_ = torch.empty((numel_,), dtype = torch.half, device = self.trellis.device)
            for n_start in range(0, self.out_features, MAX_RECONSTRUCT_SLICE_N):
                n_end = min(n_start + MAX_RECONSTRUCT_SLICE_N, self.out_features)
                numel = self.in_features * (n_end - n_start)
                w = w_[:numel].view(self.in_features, n_end - n_start)
                if use_fused:
                    ext.reconstruct_had_slice(
                        w, self.trellis, self.suh, self.svh[n_start:], self.K, self.mcg, self.mul1, n_start)
                else:
                    self._recon_slice(w, n_start)
                ext.hgemm_recon(xh, w, y_[:, n_start:n_end])

        if not use_fused:
            ext.had_r_128(y_, y_, None, self.svh, 1.0)


    def _int8_eligible(self) -> bool:
        # All N are eligible: out_features > MAX_RECONSTRUCT_SLICE_N (lm_head)
        # runs the sliced int8 path with one shared activation quant.
        # The int8 GEMM wants at least 16-aligned dims (always true here).
        return (
            self.in_features % 16 == 0 and self.out_features % 16 == 0
            and self.trellis.is_cuda
        )


    def _recon(self, w):
        ext.reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)

    def _recon_slice(self, w, n_start):
        ext.reconstruct_slice(w, self.trellis, self.K, self.mcg, self.mul1, n_start)

    def _emu_w(self, x, out_dtype):
        shape = x.shape
        rows = x.numel() // shape[-1]
        x = x.view(rows, self.in_features).contiguous()
        y = torch.empty(shape[:-1] + (self.out_features,), dtype = out_dtype or self.default_out_dtype, device = x.device)
        y_ = y.view(rows, self.out_features)
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, self.suh, None, 1.0)
        w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
        self._recon(w)
        with torch.inference_mode():
            wf = w.float()
            if not _emu_noround:
                wf = (wf / INT8_FIXED_S0).round().clamp(-128, 127) * INT8_FIXED_S0
            y_.copy_((xh.float() @ wf).to(y_.dtype))
        ext.had_r_128(y_, y_, None, self.svh, 1.0)
        if self.bias is not None:
            y += self.bias
        return y


    def _quant_w8(self, w):
        # INT8-exact models (s0 stored): the decoded tiles are exactly q * s0,
        # so int8 is one exact division with no per-call amax. Old models keep
        # the per-call per-output-channel amax grid (deliberately NOT folded
        # into an fp16 svh copy: that rounding cost +0.058 macro KLD).
        with torch.inference_mode():
            if self.s0 is not None:
                s0 = self.s0_value
                w8 = (w.float() / s0).round().clamp(-128, 127).to(torch.int8)
                return w8, s0
            # fp32 scales: a half-rounded scale would add needless error.
            s = w.float().abs().amax(dim=0).clamp_min(1e-12) / 127.0
            w8 = (w.float() / s).round().clamp(-128, 127).to(torch.int8)
        return w8, s


    def _reconstruct_hgemm_int8(self, x, y, y_, rows, out_dtype):
        # Rotated-basis W8A8: input Hadamard, int8 GEMM, output Hadamard.
        # C4 routing policy (returns None -> fp16 path): short rows stay fp16
        # (decode-bound, int8 regressed 0.56x at 512); tiny-N linears stay
        # fp16 (quant overhead dominates there). Everything else takes int8:
        # igemm beats hgemm ~4x on GEMM level because production hgemm runs
        # with fp32 accumulation output on the big linears (hg32 ~3x slower
        # than hg16, measured per shape), while igemm accumulates int32.
        if rows < _int8_min_rows or self.out_features <= 2048:
            return None
        is_head = self.out_features > MAX_RECONSTRUCT_SLICE_N
        if (_int8_scope == "head" and not is_head) or (_int8_scope == "nohead" and is_head):
            return None
        # Vocab-sized N (lm_head) takes the sliced path with one shared
        # activation quant; ablation modes decline it (fall back to fp16).
        if self.out_features > MAX_RECONSTRUCT_SLICE_N:
            if _int8_ablate:
                return None
            return self._reconstruct_hgemm_int8_sliced(x, y, y_, rows, out_dtype)
        if (not _int8_ablate and _dense_triton and self.s0 is not None
                and getattr(ext, "igemm_recon", None) is not None):
            return self._reconstruct_hgemm_int8_fast(x, y, y_, rows, out_dtype)
        key = (rows, self.in_features, self.out_features)
        if key not in _int8_logged_shapes:
            _int8_logged_shapes.add(key)
            print(f"EXL3_INT8_PREFILL engaged: rows={rows} "
                  f"k={self.in_features} n={self.out_features} "
                  f"backend={'igemm' if getattr(ext, 'igemm_recon', None) else 'emu'}",
                  flush=True)
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, self.suh, None, 1.0)
        w = torch.empty((self.in_features, self.out_features),
                        dtype=torch.half, device=self.trellis.device)
        self._recon(w)
        w8, ws = self._quant_w8(w)
        if _int8_ablate == "w":
            # Weight-grid error only: fp32 activations, fp32 matmul on the
            # dequantized int8 weights, original svh.
            with torch.inference_mode():
                y_.copy_((xh.float() @ (w8.float() * ws)).to(y_.dtype))
            del w
            ext.had_r_128(y_, y_, None, self.svh, 1.0)
            if self.bias is not None:
                y += self.bias
            return y
        if _int8_ablate == "a":
            # Activation-grid error only: fp16 weights, fp32 matmul on the
            # dequantized int8 activations, original svh.
            with torch.inference_mode():
                sa = xh.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
                a8 = (xh.float() / sa).round().clamp(-128, 127).to(torch.int8)
                y_.copy_(((a8.float() * sa) @ w.float()).to(y_.dtype))
            del w
            ext.had_r_128(y_, y_, None, self.svh, 1.0)
            if self.bias is not None:
                y += self.bias
            return y
        del w
        with torch.inference_mode():
            sa = xh.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
            a8 = (xh.float() / sa).round().clamp(-128, 127).to(torch.int8)
        igemm = getattr(ext, "igemm_recon", None)
        if igemm is not None:
            c = torch.empty((rows, self.out_features), dtype=torch.int32, device=x.device)
            igemm(a8, w8, c)
        else:
            # Exact emulation until the backend provides the int8 GEMM entry.
            c = _exact_int32_gemm(a8, w8)
        with torch.inference_mode():
            # Weight scales in fp32 (ws), output Hadamard with the ORIGINAL
            # svh: folding scales into an fp16 svh copy cost +0.058 macro KLD.
            y_.copy_((c.float() * sa * ws).to(y_.dtype))
        del c
        ext.had_r_128(y_, y_, None, self.svh, 1.0)
        if self.bias is not None:
            y += self.bias
        return y


    def _reconstruct_hgemm_int8_fast(self, x, y, y_, rows, out_dtype):
        # C2 fast path: same math as the torch reference below, but the
        # multi-pass torch quant/epilogue chains are replaced by single-pass
        # fused Triton kernels (dquant_rowwise, quant_w8_slab,
        # dense_epilog_had). Requires an s0 (int8-grid) model.
        key = (rows, self.in_features, self.out_features)
        if key not in _int8_logged_shapes:
            _int8_logged_shapes.add(key)
            print(f"EXL3_INT8_PREFILL engaged (fast): rows={rows} "
                  f"k={self.in_features} n={self.out_features} backend=igemm",
                  flush=True)
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, self.suh, None, 1.0)
        with torch.inference_mode():
            sa = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
            a8 = torch.empty((rows, self.in_features), dtype=torch.int8,
                             device=x.device)
            dquant_rowwise(xh, sa, a8)
        del xh
        w = torch.empty((self.in_features, self.out_features),
                        dtype=torch.half, device=self.trellis.device)
        self._recon(w)
        w8 = torch.empty((1, self.in_features, self.out_features),
                         dtype=torch.int8, device=w.device)
        s0t = self.s0.to(dtype=torch.float32, device=w.device)
        with torch.inference_mode():
            quant_w8_slab(w.unsqueeze(0), s0t, w8)
        del w
        c = torch.empty((rows, self.out_features), dtype=torch.int32, device=x.device)
        ext.igemm_recon(a8, w8.squeeze(0), c)
        del a8, w8
        with torch.inference_mode():
            dense_epilog_had(c, sa, s0t.to(device=x.device), self.svh, y_,
                             self.out_features)
        del c, sa
        if self.bias is not None:
            y += self.bias
        return y


    def _reconstruct_hgemm_int8_sliced(self, x, y, y_, rows, out_dtype):
        # Sliced-N variant of the int8 path (lm_head, vocab-sized N): one
        # shared input Hadamard + activation quant, then per-slice decode,
        # igemm and fused epilogue. Weight scales in fp32; output Hadamard
        # uses the original svh. Without the fused kernels (or s0 model)
        # falls back to the torch reference chain.
        fast = (not _int8_ablate and _dense_triton and self.s0 is not None
                and getattr(ext, "igemm_recon", None) is not None)
        key = (rows, self.in_features, self.out_features)
        if key not in _int8_logged_shapes:
            _int8_logged_shapes.add(key)
            print(f"EXL3_INT8_PREFILL engaged (sliced{'/fast' if fast else ''}): "
                  f"rows={rows} k={self.in_features} n={self.out_features} "
                  f"backend={'igemm' if getattr(ext, 'igemm_recon', None) else 'emu'}",
                  flush=True)
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, self.suh, None, 1.0)
        with torch.inference_mode():
            sa = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
            a8 = torch.empty((rows, self.in_features), dtype=torch.int8,
                             device=x.device)
            if fast:
                dquant_rowwise(xh, sa, a8)
            else:
                sa.copy_(xh.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0)
                a8.copy_((xh.float() / sa).round().clamp(-128, 127).to(torch.int8))
        del xh
        igemm = getattr(ext, "igemm_recon", None)
        s0t = self.s0.to(dtype=torch.float32, device=x.device) if self.s0 is not None else None
        step = MAX_RECONSTRUCT_SLICE_N
        for n_start in range(0, self.out_features, step):
            n_end = min(n_start + step, self.out_features)
            nw = n_end - n_start
            w = torch.empty((self.in_features, nw),
                            dtype=torch.half, device=self.trellis.device)
            self._recon_slice(w, n_start)
            if fast:
                w8 = torch.empty((1, self.in_features, nw), dtype=torch.int8,
                                 device=w.device)
                with torch.inference_mode():
                    quant_w8_slab(w.unsqueeze(0), s0t.to(device=w.device), w8)
                ws = s0t
                w8 = w8.squeeze(0)
            else:
                w8, ws = self._quant_w8(w)
            del w
            if igemm is not None:
                c = torch.empty((rows, nw), dtype=torch.int32, device=x.device)
                igemm(a8, w8, c)
            else:
                # Exact emulation until the backend provides the int8 GEMM entry.
                c = _exact_int32_gemm(a8, w8)
            with torch.inference_mode():
                if fast:
                    dense_epilog_had(c, sa, ws, self.svh[n_start:n_end], y_,
                                     self.out_features, n_start)
                else:
                    y_[:, n_start:n_end].copy_((c.float() * sa * ws).to(y_.dtype))
            del c, w8
        del a8
        if not fast:
            ext.had_r_128(y_, y_, None, self.svh, 1.0)
        if self.bias is not None:
            y += self.bias
        return y


    def _wscales(self, w):
        # Recompute the per-channel weight scales (ablation path only).
        with torch.inference_mode():
            return w.float().abs().amax(dim=0).clamp_min(1e-12) / 127.0


    def get_inner_weight_tensor(self, n_offset: int = 0, n_features: int | None = None):
        w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
        self._recon(w)
        if self.s0 is not None:
            # INT8-exact models: the states were chosen for the rounded grid
            # values, so decode must round (a stock decode reads measurably
            # worse weights). Values are exactly on-grid; this is idempotent.
            with torch.inference_mode():
                s0 = self.s0_value
                w = torch.round(w.float() / s0).clamp(-128, 127).mul_(s0).to(torch.half)
        return w


    def get_weight_tensor(self):
        # suh = self.unpack_bf(self.su).unsqueeze(1)
        suh = self.unpack_bf(self.su).unsqueeze(1) if self.su else self.suh.unsqueeze(1)
        svh = self.unpack_bf(self.sv).unsqueeze(0) if self.sv else self.svh.unsqueeze(0)
        w = self.get_inner_weight_tensor()
        w = preapply_had_l(w, had_k)
        w *= suh
        w = preapply_had_r(w, had_n)
        w *= svh
        # w *= self.scale
        return w


    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias


    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.trellis.device
        if self.su is not None: self.su = self.su.cpu()
        if self.sv is not None: self.sv = self.sv.cpu()
        if self.suh is not None: self.suh = self.suh.cpu()
        if self.svh is not None: self.svh = self.svh.cpu()
        if self.trellis is not None: self.trellis = self.trellis.cpu()
        if self.bias is not None: self.bias = self.bias.cpu()


    def unswap_cpu(self):
        if self.swap_device is None:
            return
        if self.su is not None: self.su = self.su.to(self.swap_device)
        if self.sv is not None: self.sv = self.sv.to(self.swap_device)
        if self.suh is not None: self.suh = self.suh.to(self.swap_device)
        if self.svh is not None: self.svh = self.svh.to(self.swap_device)
        if self.trellis is not None: self.trellis = self.trellis.to(self.swap_device)
        if self.bias is not None: self.bias = self.bias.to(self.swap_device)
        self.swap_device = None


    def tp_export(self, plan, producer):
        return {
            "cls": LinearEXL3,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "suh": producer.send(self.suh),
            "svh": producer.send(self.svh),
            "trellis": producer.send(self.trellis),
            "bias": producer.send(self.bias),
            "mcg": producer.send(self.mcg_tensor),
            "mul1": producer.send(self.mul1_tensor),
            "s0": producer.send(self.s0),
            "out_dtype": self.out_dtype,
        }


    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)
        s0 = consumer.recv(exported["s0"], cuda = True)

        if split is not None:
            split_out, first, last = split
        else:
            split_out, first, last = True, 0, exported["out_features"]

        if split_out:
            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features = last - first
        else:
            suh = consumer.recv(id_suh, cuda = True, slice_dim = 0, first = first, last = last)
            svh = consumer.recv(id_svh, cuda = True)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 0, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True) if first == 0 else None
            in_features = last - first
            out_features = exported["out_features"]

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
            s0 = s0,
        )
        return module


    @staticmethod
    def tp_import_split_3(local_context, exported, plan, split_0, split_1, split_2, dbg = False):
        return LinearEXL3.tp_import_split_n(local_context, exported, plan, [split_0, split_1, split_2], dbg)


    @staticmethod
    def tp_import_split_n(local_context, exported, plan, splits, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)
        s0 = consumer.recv(exported["s0"], cuda = True)

        svh_ = []
        trellis_ = []
        bias_ = []
        in_features = 0
        out_features = 0

        for split in splits:
            assert split is not None
            split_out, first, last = split
            assert split_out

            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features += last - first
            svh_.append(svh)
            trellis_.append(trellis)
            bias_.append(bias)

        svh = torch.cat(svh_, dim = 0)
        trellis = torch.cat(trellis_, dim = 1)
        bias = torch.cat(bias_, dim = 0) if bias_[0] is not None else None

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
            s0 = s0,
        )
        return module
