import argparse
import json
import math
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
        if experts and (experts / "expert-0511.json").exists():
            counts = [json.loads((experts / f"expert-{i:04d}.json").read_bytes())["counts"]["routed_tokens"]
                      for i in range(512)]
        else:
            counts = [1] * 512
        layers.append(counts)
    return layers


def assign(capacity, speed, counts):
    if len(capacity) != 3 or min(capacity) < 0 or len(speed) != 3 or min(speed) <= 0:
        raise ValueError("3 rank の空き容量と帯域が必要です")
    slots = [value // EXPERT_BYTES for value in capacity]
    total = len(counts) * 512
    if sum(slots) < total:
        raise MemoryError(f"expert 用の枠が不足しています: {sum(slots)} / {total}")
    quota = [0, 0, 0]
    for _ in range(total):
        rank = min((r for r in range(3) if quota[r] < slots[r]),
                   key=lambda r: (quota[r] + 1) / speed[r])
        quota[rank] += 1
    result = []
    for layer, histogram in enumerate(counts):
        left = len(counts) - layer
        limits = [q // left for q in quota]
        remainder_order = sorted(range(3), key=lambda r: quota[r] % left, reverse=True)
        for r in remainder_order[:512-sum(limits)]:
            limits[r] += 1
        if len(histogram) != 512 or min(histogram) < 0 or sum(limits) != 512:
            raise ValueError("expert の頻度または層別の配置枠が不正です")
        assigned = [-1] * 512
        load = [0, 0, 0]
        for expert in sorted(range(512), key=lambda i: -histogram[i]):
            cost = max(histogram[expert], 1)
            rank = min((r for r in range(3) if limits[r]),
                       key=lambda r: (load[r] + cost) / speed[r])
            assigned[expert] = rank
            load[rank] += cost
            limits[rank] -= 1
            quota[rank] -= 1
        result.append(assigned)
    return result


def fixed_estimate(model, work):
    import struct

    index = json.loads((model / "model.safetensors.index.json").read_bytes())["weight_map"]
    captured = {}
    for representative in (0, 1, 2):
        records = (work / f"layer-{representative:02d}/linears").glob("linear-*.json")
        captured[representative] = {json.loads(path.read_bytes())["source"].split(".",3)[3] for path in records}
        if not captured[representative]:
            raise ValueError("先頭 3 層の線形情報が必要です")
    cfg = json.loads((model / "config.json").read_bytes())["text_config"]
    fixed = draft = 0
    for shard in set(index.values()):
        with (model / shard).open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for name, spec in header.items():
            if name == "__metadata__" or name.startswith("vit.") or ".mlp.experts." in name:
                continue
            if name.endswith(("multi_head_embedding.embedding.weight", "engram_layer.lookup_table")):
                continue
            size = spec["data_offsets"][1] - spec["data_offsets"][0]
            quant = name == "lm_head.weight"
            if name.startswith("model.layers."):
                _, _, layer, suffix = name.split(".", 3)
                i = int(layer)
                source = 0 if i == 0 else (2 if cfg["layer_types"][i] == "sliding_attention" else 1)
                quant = suffix in captured[source] or (suffix.startswith("engram_layer.") and suffix in captured[1])
            is_draft = name.startswith("model.dfly_layers.")
            if is_draft:
                quant = name.endswith(".weight") and len(spec["shape"]) == 2
            if quant:
                n, k = [math.ceil(v / 128) * 128 for v in spec["shape"]]
                bits = 6 if name == "lm_head.weight" else 4
                size = n * k * bits // 8 + 2 * (n + k)
            if is_draft:
                draft += size
            else:
                fixed += size
    return fixed, draft


def main():
    p = argparse.ArgumentParser(description="原本のヘッダーと校正頻度から EP 配置と容量を試算する")
    p.add_argument("model", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("--capacity", type=float, nargs=3, default=[32,32,32])
    p.add_argument("--speed", type=float, nargs=3, default=[640,512,512])
    p.add_argument("--workspace", type=float, nargs=3, default=[2,0.5,0.5])
    p.add_argument("--tokens", type=int, default=2097152)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not 0 < args.tokens <= 2097152:
        raise ValueError("KV の予約長が範囲外です")
    fixed, draft = fixed_estimate(args.model, args.work)
    kv, indexer = cache_bytes(args.tokens)
    capacities = [int((c - w) * GIB) for c,w in zip(args.capacity, args.workspace)]
    capacities[0] -= fixed + draft + kv + indexer
    owners = assign(capacities, args.speed, routing_counts(args.work))
    rank_experts = [sum(row.count(r) for row in owners) for r in range(3)]
    print(json.dumps(dict(fixed_bytes=fixed, draft_bytes=draft, kv_bytes=kv, index_bytes=indexer,
                          rank_experts=rank_experts, expert_bytes=EXPERT_BYTES,
                          free_after=[b-count*EXPERT_BYTES for b,count in zip(capacities, rank_experts)])))
    if args.output:
        with args.output.open("x") as f:
            json.dump(owners, f)


if __name__ == "__main__":
    main()
