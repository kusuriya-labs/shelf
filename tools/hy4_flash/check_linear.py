import argparse
import ctypes
import statistics
from pathlib import Path

import torch
from safetensors.torch import load_file

from linear import HipLinear, Projection


def main():
    parser = argparse.ArgumentParser(description="実 trellis で HIP 出力と CPU 基準値を比較する")
    parser.add_argument("shard", type=Path)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--timing", type=int, default=0)
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--wmma", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    weights = load_file(args.shard)
    tensors = [weights[args.prefix + name] for name in ("trellis", "suh", "svh")]
    marker = int(weights[args.prefix + "mul1"].item())
    width = args.width or tensors[2].numel()
    x = torch.randn(args.rows, args.input_width or tensors[1].numel(),
                    generator=torch.Generator().manual_seed(123)).bfloat16()
    expected = Projection(*tensors, marker, width).cpu(x)
    if not expected.isfinite().all():
        raise ValueError("CPU 出力が非有限です")
    print("CPU 出力 RMS", expected.float().square().mean().sqrt().item())
    if args.library:
        ops = HipLinear(args.library, args.wmma)
        weight = Projection(*(t.cuda() for t in tensors), marker, width)
        gpu_input = x.cuda()
        if args.grouped:
            if weight.bits != 4 or x.size(1) != 1536 or width != 1536:
                raise ValueError("grouped は 1536×1536 の 4 bit expert 専用です")
            proj = torch.tensor([[weight.suh.data_ptr(), weight.svh.data_ptr(), weight.trellis.data_ptr()]]
                                * args.rows, device="cuda",dtype=torch.int64)
            group = ops.library.shelf_hy4_expert_linear
            group.argtypes = [ctypes.c_void_p]*2+[ctypes.c_int]+[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p]
            group.restype = ctypes.c_int
            codes = torch.empty_like(gpu_input,dtype=torch.int8)
            scales = torch.empty(args.rows,12,device="cuda",dtype=torch.float32)
            partial = torch.empty_like(gpu_input,dtype=torch.float32)
            output = torch.empty_like(gpu_input)
            def launch():
                status = group(gpu_input.data_ptr(),proj.data_ptr(),1,codes.data_ptr(),scales.data_ptr(),
                               partial.data_ptr(),output.data_ptr(),args.rows,None)
                if status:
                    raise RuntimeError(f"grouped HIP 線形の起動失敗: {status}")
                return output
        else:
            def launch():
                return ops(weight,gpu_input)
        actual = launch().cpu()
        error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
        print("HIP 出力の相対 L2", error.item())
        if not error < 0.002:
            raise ValueError("CPU 基準値との誤差が大きすぎます")
        if args.timing:
            samples = []
            for _ in range(args.timing):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                launch()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            print("HIP event 中央値 ms", statistics.median(samples), "測定回数",len(samples))


if __name__ == "__main__":
    main()
