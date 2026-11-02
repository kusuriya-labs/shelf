import json

import torch
from safetensors.torch import load_file, save_file

from quark_psq5.codebook import CB28, dequantize_weight
from quark_psq5.scale_codec import make_scale_codec

FORMAT_NAME = "PSQ5-CB28"
FORMAT_VERSION = 1
MAGNITUDE_BYTES = 16
SIGN_BYTES = 4


def stream_keys(tensor_name: str) -> dict:
    return {
        "magnitude": f"{tensor_name}.weight.psq5.magnitude",
        "sign": f"{tensor_name}.weight.psq5.sign",
        "scale": f"{tensor_name}.weight.psq5.scale",
    }


def build_manifest(entries: dict, config, codec) -> dict:
    bytes_per_block = MAGNITUDE_BYTES + SIGN_BYTES + codec.storage_bytes
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "block_k": config.block_k,
        "codebook": list(CB28),
        "code": {"bits": 5, "sign_bit": 4, "magnitude_bits": 4},
        "magnitude_layout": "4bit index per element, low nibble = even element",
        "sign_layout": "bit i of the little-endian 32-bit word = element i",
        "scale_format": codec.name,
        "scale_fold": codec.fold,
        "scale_storage": str(codec.storage_dtype).replace("torch.", ""),
        "bytes_per_block": bytes_per_block,
        "bits_per_weight": bytes_per_block * 8 / 32,
        "tensors": entries,
    }


def write_manifest(path: str, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_artifact(path: str, streams: dict, manifest: dict) -> None:
    save_file({key: value.detach().to("cpu").contiguous() for key, value in streams.items()}, path)
    write_manifest(path + ".json", manifest)


def read_artifact(path: str) -> tuple[dict, dict]:
    with open(path + ".json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return load_file(path), manifest


def reconstruct(streams: dict, manifest: dict, tensor_name: str) -> torch.Tensor:
    entry = manifest["tensors"][tensor_name]
    keys = stream_keys(tensor_name)
    codec = make_scale_codec(manifest["scale_format"], manifest["scale_fold"])
    rows, columns = entry["weight_shape"]
    return dequantize_weight(
        streams[keys["magnitude"]],
        streams[keys["sign"]],
        streams[keys["scale"]],
        codec,
        rows,
        columns,
        entry["block_k"],
    )
