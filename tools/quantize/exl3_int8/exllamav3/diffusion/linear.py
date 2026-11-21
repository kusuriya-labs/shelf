"""EXL3 weights with native BF16/FP16 activation GEMM for diffusion models."""

import torch
from torch import nn


class EXL3DiffusionLinear(nn.Module):
    """Reconstruct one bounded column slice, then multiply in the input dtype.

    The EXL3 reconstruction kernel emits FP16 weights. BF16 inputs are never
    cast to FP16: the reconstructed slice is converted to BF16 instead. This
    preserves BF16 activation range. No dense weight is retained between calls.
    ``workspace_bytes`` bounds FP16 plus converted-weight buffers, excluding
    activations, output and the BLAS library's own workspace.
    """

    def __init__(self, in_features, out_features, trellis, suh, svh, bias=None,
                 *, mcg=False, mul1=False, workspace_bytes=256 * 1024**2):
        super().__init__()
        if min(in_features, out_features) <= 0 or in_features % 128 or out_features % 128:
            raise ValueError("EXL3 diffusion dimensions must be positive multiples of 128")
        if trellis.dtype != torch.int16 or trellis.ndim != 3:
            raise ValueError("trellis must be a rank-3 int16 tensor")
        bits = trellis.shape[-1] // 16
        if not 1 <= bits <= 8 or tuple(trellis.shape) != (in_features // 16, out_features // 16, bits * 16):
            raise ValueError("trellis shape does not match dimensions and EXL3 bitrate")
        if suh.shape != (in_features,) or svh.shape != (out_features,):
            raise ValueError("EXL3 scales do not match dimensions")
        if suh.dtype != torch.float16 or svh.dtype != torch.float16:
            raise ValueError("EXL3 scale buffers must be FP16")
        if bias is not None and bias.shape != (out_features,):
            raise ValueError("bias shape does not match out_features")
        if mcg and mul1:
            raise ValueError("EXL3 codebook flags are mutually exclusive")
        if workspace_bytes < 4 * in_features * 128:
            raise ValueError("workspace must hold one 128-column FP16 and BF16 weight slice")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.mcg = bool(mcg)
        self.mul1 = bool(mul1)
        self.workspace_bytes = int(workspace_bytes)
        self.register_buffer("trellis", trellis.contiguous())
        self.register_buffer("suh", suh.contiguous())
        self.register_buffer("svh", svh.contiguous())
        self.register_buffer("bias", bias)

    def _apply(self, fn, recurse=True):
        # Pipeline .to(dtype=BF16) must not round the EXL3 FP16 scales through
        # BF16. Probe the requested destination, and move format buffers while
        # preserving their original dtype and values.
        preserved = {}
        for name in ("trellis", "suh", "svh"):
            tensor = self._buffers.pop(name)
            try:
                destination = fn(torch.empty(0, device=tensor.device, dtype=tensor.dtype))
                preserved[name] = tensor.to(device=destination.device)
            finally:
                self._buffers[name] = tensor
        original = {name: self._buffers.pop(name) for name in preserved}
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            self._buffers.update(original)
        self._buffers.update(preserved)
        return self

    def forward(self, x):
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("EXL3 diffusion activations must be FP16 or BF16")
        if x.device.type != "cuda" or x.device != self.trellis.device:
            raise ValueError("input and EXL3 buffers must be on the same GPU")
        if x.shape[-1] != self.in_features:
            raise ValueError("input width does not match EXL3 in_features")
        from ..ext import exllamav3_ext as ext

        rows = x.reshape(-1, self.in_features)
        result = torch.empty((rows.shape[0], self.out_features), device=x.device, dtype=x.dtype)
        bytes_per_weight = 2 if x.dtype == torch.float16 else 4
        slice_n = min(self.out_features,
                      (self.workspace_bytes // (bytes_per_weight * self.in_features) // 128) * 128)
        weight_fp16 = torch.empty((self.in_features, slice_n), device=x.device, dtype=torch.float16)
        weight_compute = (weight_fp16 if x.dtype == torch.float16 else torch.empty_like(weight_fp16, dtype=x.dtype))
        # The final short slice needs a contiguous storage view, not a strided
        # subset of columns, because reconstruct_had_slice writes dense rows.
        for start in range(0, self.out_features, slice_n):
            count = min(slice_n, self.out_features - start)
            w16 = weight_fp16.view(-1)[:self.in_features * count].view(self.in_features, count)
            wc = weight_compute.view(-1)[:self.in_features * count].view(self.in_features, count)
            ext.reconstruct_had_slice(w16, self.trellis, self.suh, self.svh[start:],
                                      self.bits, self.mcg, self.mul1, start)
            if x.dtype != torch.float16:
                wc.copy_(w16)
            torch.mm(rows, wc, out=result[:, start:start + count])
        if self.bias is not None:
            result.add_(self.bias)
        return result.view(*x.shape[:-1], self.out_features)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bits={self.bits}, workspace_bytes={self.workspace_bytes}")
