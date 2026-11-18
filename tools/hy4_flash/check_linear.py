import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from linear import HipLinear, Projection


def main():
    parser = argparse.ArgumentParser(description="実 trellis で HIP 出力と CPU 基準値を比較する")
    parser.add_argument("shard", type=Path)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--library", type=Path)
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
        actual = HipLinear(args.library, args.wmma)(Projection(*(t.cuda() for t in tensors), marker, width), x.cuda()).cpu()
        error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
        print("HIP 出力の相対 L2", error.item())
        if not error < 0.002:
            raise ValueError("CPU 基準値との誤差が大きすぎます")


if __name__ == "__main__":
    main()
