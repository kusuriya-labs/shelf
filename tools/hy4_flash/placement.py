import argparse
import json
from pathlib import Path


EXPERT_BYTES = 3 * (1536 * 1536 // 2 + 2 * 1536 * 2)
GIB = 1 << 30


def cache_bytes(tokens):
    sliding = min(tokens, 518)
    return 294 * (30 * sliding + 12 * tokens), 12 * (tokens // 4 + 12) * 128 * 2


def routing_counts(work):
    layers = []
    for layer in range(1, 42):
        experts = work / f"layer-{layer:02d}/experts" if work else None
        if experts:
            counts = [json.loads((experts / f"expert-{i:04d}.json").read_bytes())["counts"]["routed_tokens"]
                      for i in range(512)]
        else:
            counts = [1] * 512
        layers.append(counts)
    return layers


def assign(capacity, speed, counts, base=None, expert_bytes=EXPERT_BYTES):
    if not 1 <= len(capacity) <= 4 or min(capacity) < 0 or len(speed) != len(capacity) or min(speed) <= 0:
        raise ValueError("1〜4 rank の空き容量と帯域が必要です")
    base = base if base is not None else [0] * len(capacity)
    if len(base) != len(capacity) or min(base) < 0:
        raise ValueError("rank ごとの固定使用量が不正です")
    slots = [value // expert_bytes for value in capacity]
    total = len(counts) * 512
    if sum(slots) < total:
        raise MemoryError(f"expert 用の枠が不足しています: {sum(slots)} / {total}")
    quota = [0] * len(capacity)
    for _ in range(total):
        rank = min((r for r in range(len(capacity)) if quota[r] < slots[r]),
                   key=lambda r: (base[r] + (quota[r] + 1) * expert_bytes, r))
        quota[rank] += 1
    result = []
    for layer, histogram in enumerate(counts):
        left = len(counts) - layer
        limits = [q // left for q in quota]
        remainder_order = sorted(range(len(capacity)), key=lambda r: quota[r] % left, reverse=True)
        for r in remainder_order[:512-sum(limits)]:
            limits[r] += 1
        if len(histogram) != 512 or min(histogram) < 0 or sum(limits) != 512:
            raise ValueError("expert の頻度または層別の配置枠が不正です")
        assigned = [-1] * 512
        load = [0] * len(capacity)
        for expert in sorted(range(512), key=lambda i: -histogram[i]):
            cost = max(histogram[expert], 1)
            rank = min((r for r in range(len(capacity)) if limits[r]),
                       key=lambda r: (load[r] + cost) / speed[r])
            assigned[expert] = rank
            load[rank] += cost
            limits[rank] -= 1
            quota[rank] -= 1
        result.append(assigned)
    return result


def checkpoint_sizes(directory):
    import struct
    import re

    index = json.loads((directory / "model.safetensors.index.json").read_bytes())["weight_map"]
    fixed = 0
    experts = {}
    for shard in set(index.values()):
        with (directory / shard).open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for name, spec in header.items():
            if name == "__metadata__" or name not in index:
                continue
            if name.endswith((".mul1", ".lookup_table", ".multi_head_embedding.embedding.weight",
                              ".multi_head_embedding.embedding.weight.scale")):
                continue
            size = spec["data_offsets"][1] - spec["data_offsets"][0]
            size = (size + 15) & ~15
            match = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.", name)
            if match:
                key = (int(match.group(1)), int(match.group(2)))
                experts[key] = experts.get(key, 0) + size
            else:
                fixed += size
    return fixed, experts


def main():
    p = argparse.ArgumentParser(description="原本のヘッダーと校正頻度から EP 配置と容量を試算する")
    p.add_argument("model", type=Path)
    p.add_argument("--work", type=Path, help="校正頻度のディレクトリ")
    p.add_argument("--draft", type=Path, help="DFly checkpoint")
    p.add_argument("--capacity", type=float, nargs="+", default=[32,32,32])
    p.add_argument("--workspace", type=float, nargs="+", default=None)
    p.add_argument("--tokens", type=int, default=2097152)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--safety-gib", type=float, default=0.5)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not 0 < args.tokens <= 2097152:
        raise ValueError("KV の予約長が範囲外です")
    if not 1 <= args.tensor_parallel <= 4:
        raise ValueError("TP の値が範囲外です")
    args.workspace = args.workspace or [2] + [0.5] * (len(args.capacity) - 1)
    if len(args.workspace) != len(args.capacity):
        raise ValueError("作業域と GPU の台数が一致しません")
    fixed, experts = checkpoint_sizes(args.model)
    if len(experts) != 41 * 512 or any(size != EXPERT_BYTES for size in experts.values()):
        raise ValueError("4 bpw expert の数または tensor サイズが不正です")
    draft = checkpoint_sizes(args.draft)[0] if args.draft else 0
    kv, indexer = cache_bytes(args.tokens)
    if args.safety_gib < 0:
        raise ValueError("安全余裕が不正です")
    config = json.loads((args.model / "config.json").read_bytes())
    storage = config["quantization_config"]["linear_storage"]
    folded = sum(meta["original_shape"][1] * 512 * 2 for name, meta in storage.items()
                 if name.endswith(".self_attn.kv_b_proj.weight"))
    matrix_bytes = 41 * 512 * 3 * 32
    expert_bytes = EXPERT_BYTES // args.tensor_parallel
    base = [int(w * GIB) + matrix_bytes for w in args.workspace]
    if args.tensor_parallel > 1:
        local = fixed // args.tensor_parallel + draft + kv + indexer + folded // args.tensor_parallel
        base = [value + local for value in base]
    else:
        base[0] += fixed + draft + kv + indexer + folded
    capacities = [int(c * GIB) - used - int(args.safety_gib * GIB)
                  for c, used in zip(args.capacity, base)]
    owners = assign(capacities, [1] * len(capacities), routing_counts(args.work), base, expert_bytes)
    rank_experts = [sum(row.count(r) for row in owners) for r in range(len(capacities))]
    print(json.dumps(dict(fixed_bytes=fixed, draft_bytes=draft, kv_bytes=kv, index_bytes=indexer,
                          folded_bytes=folded, matrix_bytes=matrix_bytes, safety_gib=args.safety_gib,
                          rank_experts=rank_experts, expert_bytes=expert_bytes,
                          peak_estimate=[b+count*expert_bytes for b,count in zip(base, rank_experts)],
                          free_after=[b-count*expert_bytes for b,count in zip(capacities, rank_experts)])))
    if args.output:
        with args.output.open("x") as f:
            json.dump(owners, f)


if __name__ == "__main__":
    main()
