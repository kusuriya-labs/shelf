"""Explicit EXL3 DiT checkpoint layout, independent of the LLM model loader."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .linear import EXL3DiffusionLinear
from .quantize import quantize_linear
from .architectures import block_linears, transformer_class, transformer_name


def _json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _tensor_file(path, tensors):
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()}, temporary)
    temporary.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def convert_transformer(model_path, destination, calibration_dirs, *, bits=4, device="cuda:0"):
    """Convert all sampled block Linear layers, with resumable per-layer files.

    A conversion manifest is only published after every target layer completes.
    Individual layer files can be resumed only with identical source/calibration
    identity. Calibration from another checkpoint is rejected.
    """
    model_path, destination = Path(model_path), Path(destination)
    config = json.loads((model_path / "transformer/config.json").read_text())
    family = transformer_name(config)
    source = json.loads((model_path / "exl3_source.json").read_text())
    calibration_dirs = [Path(p) for p in calibration_dirs]
    captures = [json.loads((p / "calibration.json").read_text()) for p in calibration_dirs]
    if not captures or any(c["provenance"]["source"] != source for c in captures):
        raise ValueError("Calibration must use this exact source model revision")
    names = sorted(set().union(*(c["layers"] for c in captures)))
    if not names or (family == "QwenImage21Transformer2DModel" and
                     any(sorted(c["layers"]) != names for c in captures)):
        raise ValueError("Calibration layer inventories disagree or are empty")
    identity = {"schema_version": 1, "source": source, "bits": bits,
                "calibration": [_hash(p / "calibration.json") for p in calibration_dirs],
                "transformer_config_sha256": _hash(model_path / "transformer/config.json")}
    if family != "QwenImage21Transformer2DModel":
        identity["transformer_type"] = family
    destination.mkdir(parents=True, exist_ok=True)
    job_path = destination / "conversion_job.json"
    if job_path.exists() and json.loads(job_path.read_text()) != identity:
        raise ValueError("Existing conversion has different source or calibration; use a new destination")
    if (destination / "quantization.json").exists():
        raise FileExistsError("A completed checkpoint already exists")
    _json(job_path, identity)
    model = transformer_class(config).from_pretrained(
        model_path / "transformer", torch_dtype=torch.bfloat16, local_files_only=True,
    ).eval()
    expected_targets = sorted(block_linears(model))
    if names != expected_targets:
        raise ValueError("Calibration does not cover all eligible block Linear layers")
    source_numel = sum(t.numel() for t in model.state_dict().values())
    source_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    records = {}
    for index, name in enumerate(names):
        layer = model.get_submodule(name)
        if not isinstance(layer, torch.nn.Linear) or name not in expected_targets:
            raise ValueError(f"Unexpected calibration target {name}")
        stem = hashlib.sha256(name.encode()).hexdigest()
        tensor_path, record_path = destination / (stem + ".safetensors"), destination / (stem + ".json")
        if record_path.exists() and tensor_path.exists():
            record = json.loads(record_path.read_text())
            if record["name"] != name or record["sha256"] != _hash(tensor_path):
                raise ValueError(f"Corrupt resumed layer {name}")
            records[name] = record
            print(f"[{index + 1}/{len(names)}] resume {name}", flush=True)
        else:
            samples = []
            for directory, capture in zip(calibration_dirs, captures):
                if name not in capture["layers"]:
                    continue
                entry = capture["layers"][name]
                if (entry["in_features"], entry["out_features"]) != (layer.in_features, layer.out_features):
                    raise ValueError(f"Calibration shape mismatch: {name}")
                samples.append(load_file(directory / entry["file"])["inputs"])
            inputs = torch.cat(samples)
            print(f"[{index + 1}/{len(names)}] quantize {name} {tuple(layer.weight.shape)}", flush=True)
            quantized, stats = quantize_linear(layer, inputs, bits=bits, device=device, seed=index)
            _tensor_file(tensor_path, quantized.state_dict())
            record = {"name": name, "file": tensor_path.name, "sha256": _hash(tensor_path),
                      "mcg": False, "mul1": False, "bias": layer.bias is not None, **stats}
            _json(record_path, record)
            records[name] = record
            del quantized, inputs, samples
        # This layer is now on disk; release its original dense CPU weights.
        model.set_submodule(name, torch.nn.Identity())
    _tensor_file(destination / "unquantized.safetensors", model.state_dict())
    _json(destination / "transformer_config.json", json.loads((model_path / "transformer/config.json").read_text()))
    manifest = {**identity, "complete": True, "layers": records,
                "source_state_numel": source_numel, "source_state_bytes": source_bytes,
                "unquantized_sha256": _hash(destination / "unquantized.safetensors"),
                "quantized_state_bytes": sum(r["quantized_state_bytes"] for r in records.values()),
                "unquantized_state_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values())}
    total_bytes = manifest["quantized_state_bytes"] + manifest["unquantized_state_bytes"]
    manifest["effective_bits_per_source_element"] = 8 * total_bytes / source_numel
    manifest["state_size_ratio"] = total_bytes / source_bytes
    _json(destination / "quantization.json", manifest)
    return manifest


def load_transformer(directory, *, device="cuda:0", workspace_bytes=256 * 1024**2):
    """Load EXL3 buffers directly without allocating original dense block weights."""
    from accelerate import init_empty_weights
    directory = Path(directory)
    manifest = json.loads((directory / "quantization.json").read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("complete"):
        raise ValueError("Unsupported or incomplete EXL3 diffusion checkpoint")
    # Keep small derived/nonpersistent buffers real (timestep and rotary
    # frequencies are not in the checkpoint); only parameters are meta.
    config = json.loads((directory / "transformer_config.json").read_text())
    if manifest.get("transformer_type", "QwenImage21Transformer2DModel") != transformer_name(config):
        raise ValueError("Checkpoint transformer family differs from configuration")
    with init_empty_weights(include_buffers=False):
        model = transformer_class(config).from_config(config)
    if sorted(manifest["layers"]) != sorted(block_linears(model)):
        raise ValueError("Checkpoint does not cover all eligible block Linear layers")
    for name, record in manifest["layers"].items():
        original = model.get_submodule(name)
        if not isinstance(original, torch.nn.Linear) or (original.in_features, original.out_features) != (record["in_features"], record["out_features"]):
            raise ValueError(f"Checkpoint configuration mismatch: {name}")
        file = directory / record["file"]
        if _hash(file) != record["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
        state = load_file(file, device=str(device))
        quantized = EXL3DiffusionLinear(record["in_features"], record["out_features"],
                                        state["trellis"], state["suh"], state["svh"], state.get("bias"),
                                        mcg=record["mcg"], mul1=record["mul1"], workspace_bytes=workspace_bytes)
        if quantized.bits != record["bits"]:
            raise ValueError(f"Checkpoint bitrate mismatch: {name}")
        model.set_submodule(name, quantized)
    file = directory / "unquantized.safetensors"
    if _hash(file) != manifest["unquantized_sha256"]:
        raise ValueError("Unquantized checkpoint checksum mismatch")
    # Quantized buffers are already resident; only the remaining parameters are
    # assigned from this file. Missing keys must correspond exactly to them.
    expected_missing = {f"{name}.{key}" for name in manifest["layers"]
                        for key in model.get_submodule(name).state_dict()}
    incompatible = model.load_state_dict(load_file(file, device=str(device)), strict=False, assign=True)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(f"Incomplete checkpoint state: {incompatible}")
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Checkpoint left uninitialized meta tensors")
    return model.to(device).eval(), manifest
