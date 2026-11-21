"""Bounded activation sampling at selected denoising steps."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from .architectures import block_linears


class ActivationCapture:
    def __init__(self, model, steps, rows_per_call=128, *, allow_uncalled=False):
        if rows_per_call <= 0 or not steps or min(steps) < 0:
            raise ValueError("Capture needs nonnegative steps and a positive row limit")
        self.steps = set(steps)
        self.rows_per_call = rows_per_call
        self.step = 0
        self.allow_uncalled = allow_uncalled
        self.samples = {}
        self.records = {}
        self.handles = []
        for name, module in block_linears(model).items():
            self.samples[name] = []
            self.records[name] = {"in_features": module.in_features,
                                  "out_features": module.out_features, "calls": []}

            def hook(mod, inputs, name=name):
                if self.step not in self.steps:
                    return
                x = inputs[0].detach().reshape(-1, mod.in_features)
                if not x.shape[0]:
                    return
                indices = torch.linspace(0, x.shape[0] - 1,
                                         min(self.rows_per_call, x.shape[0]),
                                         device=x.device).long()
                sample = x.index_select(0, indices).cpu()
                if not torch.isfinite(sample).all():
                    raise ValueError(f"Nonfinite calibration activation in {name} at step {self.step}")
                self.samples[name].append(sample)
                self.records[name]["calls"].append({
                    "step": self.step, "input_rows": x.shape[0], "sample_rows": sample.shape[0],
                    "dtype": str(sample.dtype), "abs_max": sample.float().abs().max().item(),
                })

            self.handles.append(module.register_forward_pre_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(self, destination, provenance):
        destination = Path(destination)
        if (destination / "calibration.json").exists():
            raise FileExistsError("Calibration manifest already exists; use a new destination")
        if not self.samples or not any(self.samples.values()) or (
                not self.allow_uncalled and any(not rows for rows in self.samples.values())):
            raise ValueError("Calibration is incomplete: some target layers have no samples")
        destination.mkdir(parents=True, exist_ok=True)
        for name, chunks in self.samples.items():
            if not chunks:
                continue
            file = hashlib.sha256(name.encode()).hexdigest() + ".safetensors"
            save_file({"inputs": torch.cat(chunks).contiguous()}, destination / file)
            self.records[name]["file"] = file
        manifest = {"schema_version": 1, "steps": sorted(self.steps), "rows_per_call": self.rows_per_call,
                    "provenance": provenance,
                    "layers": {name: entry for name, entry in self.records.items() if self.samples[name]}}
        (destination / "calibration.json").write_text(json.dumps(manifest, indent=2) + "\n")
