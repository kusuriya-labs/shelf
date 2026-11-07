#!/usr/bin/env python3
"""EXL3 HF checkpoint header analyzer.

Reads config.json, quantization_config.json and safetensors headers without
mapping the weight payloads, then emits a per-tensor manifest and a summary.

The analyzer understands the Qwen4Exp PLE n-gram table layout
(exl3_ngram_trellis) and the main EXL3 trellis modules.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

DTYPE_SIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

EXL3_SUFFIXES = ("trellis", "suh", "svh", "mul1", "mcg")

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def dtype_size(dtype: str) -> int:
    if dtype not in DTYPE_SIZE:
        raise ValueError(f"unsupported safetensors dtype: {dtype}")
    return DTYPE_SIZE[dtype]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_safetensors_header(path: Path):
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated header length")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len == 0 or header_len > size - 8:
            raise ValueError(f"{path}: invalid header length {header_len}")
        payload = handle.read(header_len)
        if len(payload) != header_len:
            raise ValueError(f"{path}: truncated header")
    try:
        header = json.loads(payload)
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: header is not UTF-8: {error}") from error
    metadata = header.pop("__metadata__", None)
    return header_len, metadata or {}, header, size


def classify(name: str):
    if ".ngram_embedding.shard_" in name:
        return "ple", "ple_embedding"
    if ".ngram_embedding." in name:
        return "ple_metadata", "ple_embedding"
    if name.startswith("mtp."):
        return "mtp", "mtp"
    if name.startswith("model.visual.") or name.startswith("visual."):
        return "vision", "vision"
    if name == "lm_head" or name.startswith("lm_head."):
        return "body", "lm_head"
    if ".ple." in name:
        return "body", "ple_aux"
    return "body", "language"


def module_of(name: str) -> str:
    base = name
    for suffix in (".weight", ".bias", ".trellis", ".suh", ".svh", ".mul1", ".mcg"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if ".shard_" in base:
        base = base.rsplit(".shard_", 1)[0]
    return base


def layer_of(name: str):
    match = _LAYER_RE.search(name)
    if match:
        return int(match.group(1))
    if name.startswith("mtp.layers."):
        match = re.search(r"mtp\.layers\.(\d+)\.", name)
        if match:
            return -1
    return None


def parse_exl3(module_record):
    trellis = module_record.get("trellis")
    if trellis is None:
        return None
    shape = trellis["shape"]
    if len(shape) != 3:
        return {"error": "trellis is not rank-3"}
    kt, nt, words = shape
    if words % 16 == 0:
        ka = words // 16
        half = False
    elif words % 16 == 8:
        ka = (words - 8) // 16
        half = True
    else:
        return {"error": "trellis words is not a valid EXL3 rate"}
    if ka < 1 or ka > 8 or (half and ka > 3):
        return {"error": "trellis bit rate out of range"}
    k = kt * 16
    n = nt * 16
    bits = ka + (0.5 if half else 0.0)
    errors = []
    suh = module_record.get("suh")
    svh = module_record.get("svh")
    mul1 = module_record.get("mul1")
    if suh is None:
        errors.append("missing suh")
    elif suh["dtype"] != "F16" or suh["shape"] != [k]:
        errors.append("suh shape/dtype mismatch")
    if svh is None:
        errors.append("missing svh")
    elif svh["dtype"] != "F16" or svh["shape"] != [n]:
        errors.append("svh shape/dtype mismatch")
    if mul1 is None:
        errors.append("missing mul1")
    elif mul1["dtype"] != "I32" or mul1["shape"] != []:
        errors.append("mul1 shape/dtype mismatch")
    if k % 128 or n % 128:
        errors.append("shape is not 128 aligned")
    return {
        "k": k,
        "n": n,
        "ka": ka,
        "half": half,
        "bits_per_weight": bits,
        "words_per_tile": words,
        "errors": errors,
    }


def find_nth_prime_after(start: int, count: int) -> int:
    def is_prime(value: int) -> bool:
        if value < 2:
            return False
        if value % 2 == 0:
            return value == 2
        limit = math.isqrt(value)
        for divisor in range(3, limit + 1, 2):
            if value % divisor == 0:
                return False
        return True

    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_MASK64 = (1 << 64) - 1
_PRIME_1 = 11400714819323198485


def splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (splitmix64(value) % half_bound) + 1)
    return multipliers


def read_tensor_bytes(path: Path, header_len: int, record) -> bytes:
    start = 8 + header_len + record["data_offsets"][0]
    length = record["data_offsets"][1] - record["data_offsets"][0]
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(length)
    if len(data) != length:
        raise ValueError(f"{path}: short read for {record}")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="directory holding config.json and safetensors files")
    parser.add_argument("--out", help="manifest.json output path")
    parser.add_argument("--summary", help="summary.json output path")
    parser.add_argument("--include-extra", action="store_true",
                        help="also scan safetensors files not listed in the index (e.g. ngram_embedding.safetensors)")
    parser.add_argument("--no-tensor-list", action="store_true",
                        help="omit the per-tensor list from manifest.json")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir).resolve()
    config = read_json(model_dir / "config.json")
    text_config = config.get("text_config", config)
    quant_config_path = model_dir / "quantization_config.json"
    quantization_config = read_json(quant_config_path) if quant_config_path.exists() else config.get("quantization_config", {})
    tensor_storage = quantization_config.get("tensor_storage", {})

    index_path = model_dir / "model.safetensors.index.json"
    index = read_json(index_path) if index_path.exists() else None
    weight_map = index.get("weight_map", {}) if index else {}
    indexed_files = sorted(set(weight_map.values()))

    files = []
    for name in indexed_files:
        path = model_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(path)
    if args.include_extra:
        for path in sorted(model_dir.glob("*.safetensors")):
            if path.name not in indexed_files:
                files.append(path)

    physical = {}
    file_records = []
    problems = []
    total_payload = 0
    for path in files:
        header_len, metadata, header, size = read_safetensors_header(path)
        data_start = 8 + header_len
        data_bytes = size - data_start
        file_records.append({
            "file": str(path),
            "size": size,
            "header_bytes": header_len,
            "data_bytes": data_bytes,
            "tensor_count": len(header),
            "metadata": metadata,
        })
        total_payload += data_bytes
        ranges = []
        for name, record in header.items():
            begin, end = record["data_offsets"]
            if begin < 0 or end < begin or end > data_bytes:
                problems.append(f"{path.name}:{name}: data_offsets outside payload")
                continue
            shape = record["shape"]
            count = 1
            for dim in shape:
                if not isinstance(dim, int) or dim < 0:
                    problems.append(f"{path.name}:{name}: invalid shape {shape}")
                    count = 0
                    break
                count *= dim
            expected = count * dtype_size(record["dtype"])
            if expected != end - begin:
                problems.append(
                    f"{path.name}:{name}: byte count {end - begin} != shape/dtype {expected}")
            if name in physical:
                problems.append(f"duplicate tensor name: {name}")
            physical[name] = {
                "name": name,
                "file": str(path),
                "file_name": path.name,
                "header_bytes": header_len,
                "data_offset": begin,
                "offset": data_start + begin,
                "nbytes": end - begin,
                "shape": shape,
                "dtype": record["dtype"],
            }
            ranges.append((begin, end, name))
        ranges.sort()
        for (begin_a, end_a, name_a), (begin_b, end_b, name_b) in zip(ranges, ranges[1:]):
            if end_a > begin_b:
                problems.append(f"{path.name}: overlapping tensors {name_a} and {name_b}")
        if data_bytes < (ranges[-1][1] if ranges else 0):
            problems.append(f"{path.name}: payload smaller than last tensor end")

    modules = collections.defaultdict(dict)
    for name, record in physical.items():
        module = module_of(name)
        kind = name.rsplit(".", 1)[-1]
        if kind in EXL3_SUFFIXES:
            modules[module][kind] = record
        elif kind in ("weight", "bias"):
            modules[module][kind] = record

    module_records = []
    tensor_records = []
    for name in sorted(physical):
        record = physical[name]
        role, category = classify(name)
        layer = layer_of(name)
        module = module_of(name)
        kind = name.rsplit(".", 1)[-1]
        storage_type = record["dtype"]
        exl3 = None
        if kind in EXL3_SUFFIXES:
            exl3 = parse_exl3(modules[module])
            if exl3 and "ka" in exl3:
                storage_type = "exl3_%s%s" % (exl3["ka"], "h" if exl3["half"] else "")
            else:
                storage_type = "exl3_unknown"
        entry = {
            "name": name,
            "file": record["file"],
            "offset": record["offset"],
            "nbytes": record["nbytes"],
            "shape": record["shape"],
            "storage_type": storage_type,
            "dtype": record["dtype"],
            "role": role,
            "category": category,
            "layer": layer,
            "module": module,
        }
        tensor_records.append(entry)

    for module, records in sorted(modules.items()):
        if "trellis" not in records:
            continue
        module_role, module_category = classify(module + ".weight")
        if module_category == "ple_embedding":
            continue
        parsed = parse_exl3(records)
        if "ka" not in parsed:
            problems.append(f"{module}: {parsed.get('error', 'invalid exl3 layout')}")
            continue
        declared = tensor_storage.get(module, {}).get("bits_per_weight")
        if declared is not None and abs(float(declared) - parsed["bits_per_weight"]) > 1e-6:
            problems.append(
                f"{module}: declared bits {declared} != trellis rate {parsed['bits_per_weight']}")
        if "verification" not in parsed:
            for error in parsed["errors"]:
                problems.append(f"{module}: {error}")
        module_records.append({
            "module": module,
            "role": module_role,
            "category": module_category,
            "layer": layer_of(module),
            "quant_format": "exl3",
            "k": parsed["k"],
            "n": parsed["n"],
            "ka": parsed["ka"],
            "half": parsed["half"],
            "bits_per_weight": parsed["bits_per_weight"],
            "trellis_bytes": records["trellis"]["nbytes"],
            "suh_bytes": records.get("suh", {}).get("nbytes", 0),
            "svh_bytes": records.get("svh", {}).get("nbytes", 0),
            "mul1_bytes": records.get("mul1", {}).get("nbytes", 0),
            "declared_bits_per_weight": tensor_storage.get(module, {}).get("bits_per_weight"),
            "errors": parsed["errors"],
        })

    totals = collections.defaultdict(int)
    counts = collections.defaultdict(int)
    for entry in tensor_records:
        key = (entry["role"], entry["category"])
        totals[key] += entry["nbytes"]
        counts[key] += 1

    totals_by_layer = collections.defaultdict(int)
    for entry in tensor_records:
        if entry["layer"] is not None and entry["role"] == "body":
            totals_by_layer[entry["layer"]] += entry["nbytes"]

    ngram_validation = {}
    ngram_tensors = [entry for entry in tensor_records if entry["role"] in ("ple", "ple_metadata")]
    if ngram_tensors:
        shard_re = re.compile(r"\.shard_(\d+)\.trellis$")
        shards = []
        for entry in ngram_tensors:
            match = shard_re.search(entry["name"])
            if match:
                shards.append((int(match.group(1)), entry))
        shards.sort()
        metadata = {}
        for file_record in file_records:
            if any("ngram_embedding" in entry["name"] and entry["file"] == file_record["file"]
                   for entry in ngram_tensors):
                metadata = file_record["metadata"]
                break
        by_name = {entry["name"].rsplit(".", 1)[-1]: entry for entry in ngram_tensors}
        aux = {}
        for suffix in ("head_bias", "head_offsets", "head_vocab_sizes", "layer_multipliers"):
            matches = [entry for entry in ngram_tensors if entry["name"].endswith("." + suffix)]
            if matches:
                entry = matches[0]
                path = Path(entry["file"])
                header_len, _, header, _ = read_safetensors_header(path)
                aux[suffix] = read_tensor_bytes(path, header_len, header[entry["name"]])
        errors = []
        if not shards:
            errors.append("no ngram shards")
        if shards:
            indices = [index for index, _ in shards]
            if indices != list(range(len(shards))):
                errors.append("ngram shard indices are not contiguous from zero")
            shapes = {tuple(entry["shape"]) for _, entry in shards}
            if len(shapes) != 1:
                errors.append("ngram shards have different shapes")
            row_words = shards[0][1]["shape"][1]
            if row_words <= 1:
                errors.append("invalid ngram row width")
            K = (row_words - 1) * 16 // 160 if row_words > 1 else None
            if K is None or 1 + 160 * K // 16 != row_words:
                errors.append("ngram row width is not a valid 160-D trellis ring")
            total_rows = sum(entry["shape"][0] for _, entry in shards)
            metadata_rows = metadata.get("rows")
            if metadata_rows is not None and int(metadata_rows) != total_rows:
                errors.append(f"ngram metadata rows {metadata_rows} != tensor rows {total_rows}")
            declared = tensor_storage.get(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding", {})
            if declared:
                declared_bytes = sum(t["n_bytes"] for t in declared.get("stored_tensors", {}).values())
                actual_bytes = sum(entry["nbytes"] for entry in ngram_tensors)
                if declared_bytes != actual_bytes:
                    errors.append(
                        f"quantization_config ngram bytes {declared_bytes} != physical {actual_bytes}")

            ngram_size = int(text_config.get("ngram_size", 0))
            heads_per_ngram = int(text_config.get("heads_per_ngram", 0))
            ple_heads = (ngram_size - 1) * heads_per_ngram
            ngram_validation["row_words"] = row_words
            ngram_validation["K"] = K
            ngram_validation["row_bytes"] = row_words * 2
            ngram_validation["total_rows"] = total_rows
            ngram_validation["shard_count"] = len(shards)
            ngram_validation["shard_rows"] = shards[0][1]["shape"][0]
            ngram_validation["effective_bpw"] = (16 + 160 * K) / 160 if K else None

            if "head_vocab_sizes" in aux:
                sizes = list(struct.unpack("<" + "q" * (len(aux["head_vocab_sizes"]) // 8),
                                           aux["head_vocab_sizes"]))
                ngram_validation["head_vocab_sizes"] = sizes
            if "head_offsets" in aux:
                offsets = list(struct.unpack("<" + "q" * (len(aux["head_offsets"]) // 8),
                                             aux["head_offsets"]))
                ngram_validation["head_offsets"] = offsets
            if "layer_multipliers" in aux:
                multipliers = list(struct.unpack("<" + "q" * (len(aux["layer_multipliers"]) // 8),
                                                 aux["layer_multipliers"]))
                ngram_validation["layer_multipliers"] = multipliers
            if "head_bias" in aux:
                ngram_validation["head_bias_shape"] = [len(aux["head_bias"]) // (2 * 160), 160]

            if "head_vocab_sizes" in ngram_validation and "head_offsets" in ngram_validation:
                sizes = ngram_validation["head_vocab_sizes"]
                offsets = ngram_validation["head_offsets"]
                if len(sizes) != ple_heads:
                    errors.append(f"head count {len(sizes)} != config derived {ple_heads}")
                if len(offsets) != len(sizes):
                    errors.append("head_offsets length != head_vocab_sizes length")
                running = 0
                for index, size in enumerate(sizes):
                    if offsets[index] != running:
                        errors.append(f"head_offsets[{index}] != cumulative {running}")
                    running += size
                padding = int(text_config.get("make_ngram_vocab_size_divisible_by", 1))
                padded = math.ceil(running / padding) * padding
                ngram_validation["total_vocab_size"] = running
                ngram_validation["padded_vocab_size"] = padded
                if padded != total_rows:
                    errors.append(f"padded ngram rows {padded} != tensor rows {total_rows}")
                base = int(text_config.get("ngram_vocab_size_base", 0))
                expected_sizes = [find_nth_prime_after(base - 1, index + 1)
                                  for index in range(ple_heads)]
                if sizes != expected_sizes:
                    errors.append("head_vocab_sizes do not match prime sequence")
                seed = int(text_config.get("seed", 1234))
                expected_multipliers = build_layer_multipliers(
                    int(text_config.get("vocab_size", 0)), ngram_size, 0, seed)
                if ngram_validation.get("layer_multipliers") != expected_multipliers:
                    errors.append("layer_multipliers do not match config-derived values")
        ngram_validation["errors"] = errors
        problems.extend(errors)

    role_totals = {}
    for (role, category), value in sorted(totals.items()):
        role_totals[f"{role}/{category}"] = {
            "tensors": counts[(role, category)],
            "bytes": value,
            "gib": value / 2**30,
        }

    body_bytes = sum(entry["nbytes"] for entry in tensor_records if entry["role"] == "body")
    text_body_bytes = sum(
        entry["nbytes"] for entry in tensor_records
        if entry["role"] == "body" and entry["category"] != "vision")
    text_body_no_mtp_bytes = sum(
        entry["nbytes"] for entry in tensor_records
        if entry["role"] == "body" and entry["category"] not in ("vision", "mtp"))
    ple_bytes = sum(entry["nbytes"] for entry in tensor_records if entry["role"] == "ple")
    ple_meta_bytes = sum(entry["nbytes"] for entry in tensor_records if entry["role"] == "ple_metadata")

    summary = {
        "model_dir": str(model_dir),
        "format": "safetensors+exl3",
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "text_model_type": text_config.get("model_type"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_experts": text_config.get("num_experts"),
        "num_experts_per_tok": text_config.get("num_experts_per_tok"),
        "vocab_size": text_config.get("vocab_size"),
        "ple_layer_ids": text_config.get("ple_layer_ids"),
        "files": file_records,
        "tensor_count": len(tensor_records),
        "payload_bytes": total_payload,
        "body_bytes": body_bytes,
        "body_gib": body_bytes / 2**30,
        "text_body_bytes": text_body_bytes,
        "text_body_gib": text_body_bytes / 2**30,
        "text_body_without_mtp_bytes": text_body_no_mtp_bytes,
        "text_body_without_mtp_gib": text_body_no_mtp_bytes / 2**30,
        "ple_bytes": ple_bytes,
        "ple_gib": ple_bytes / 2**30,
        "ple_metadata_bytes": ple_meta_bytes,
        "totals_by_role": role_totals,
        "totals_by_layer": {str(k): v for k, v in sorted(totals_by_layer.items())},
        "module_count": len(modules),
        "exl3_module_count": len(module_records),
        "exl3_bits_histogram": dict(collections.Counter(
            record["bits_per_weight"] for record in module_records)),
        "exl3_bits_histogram_by_category": {
            category: dict(collections.Counter(
                record["bits_per_weight"] for record in module_records
                if record["category"] == category))
            for category in sorted({record["category"] for record in module_records})
        },
        "exl3_modules_with_errors": sum(1 for record in module_records if record["errors"]),
        "ngram": ngram_validation,
        "problems": problems,
    }

    manifest = {
        "model_dir": str(model_dir),
        "generated_by": "tools/analyze_exl3_hf_manifest.py",
        "files": file_records,
        "summary": {key: value for key, value in summary.items() if key not in ("files", "problems")},
        "tensors": [] if args.no_tensor_list else tensor_records,
        "modules": module_records,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, separators=(",", ":"))
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1, ensure_ascii=False)

    print(json.dumps({
        "tensor_count": len(tensor_records),
        "module_count": len(modules),
        "exl3_module_count": len(module_records),
        "body_gib": round(body_bytes / 2**30, 3),
        "ple_gib": round(ple_bytes / 2**30, 3),
        "exl3_bits_histogram": summary["exl3_bits_histogram"],
        "ngram": {k: v for k, v in ngram_validation.items() if k not in ("head_offsets",)},
        "problem_count": len(problems),
        "problems": problems[:50],
    }, indent=1, ensure_ascii=False))

    if problems:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
