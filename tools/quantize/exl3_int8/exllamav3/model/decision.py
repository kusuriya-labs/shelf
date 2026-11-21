"""JEV bare-v1 typed decisions, exact head packaging and calibrated readout.

Prompt/readout follows autotrust/JEV-27B-VL serve_decide.py. The batch wire
shape follows llama.cpp tools/server/server-decision.cpp (7049ff0c).
No generation, sampling filters or vocabulary softmax enters System 1.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import shutil
import string
import os


class DecisionProfile:
    def __init__(self, head, calibration, encode):
        self.head = head
        slots = head['slots']
        if slots['template_version'] != 'bare-v1':
            raise ValueError('Only JEV bare-v1 decision adapters are supported')
        self.ranges = slots['ranges']
        self.ids = head['verbalizer_ids']
        self.bias = head['bias']
        if len(self.ids) != slots['num_slots'] or len(self.bias) != len(self.ids):
            raise ValueError('Invalid decision-head slots')
        self.temperatures = calibration['per_kind']
        for kind in ('noul', 'score', 'choice'):
            lo, hi = self.ranges[kind]
            if not 0 <= lo < hi <= len(self.ids):
                raise ValueError('Invalid decision slot range')
            temp = self.temperatures[kind]
            if not math.isfinite(temp) or temp <= 0:
                raise ValueError('Decision temperature must be finite and positive')
        if not all(math.isfinite(x) for x in self.bias):
            raise ValueError('Non-finite decision bias')
        self.labels, self.label_ids = [], []
        codes = list(string.ascii_uppercase) + [a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        for code in codes:
            ids = encode(code)
            if len(ids) == 1 and ids[0] in encode(f'x\n{code}) y'):
                self.labels.append(code)
                self.label_ids.append(ids[0])
                if len(self.labels) == 256:
                    break
        lo, hi = self.ranges['choice']
        if self.label_ids[:hi-lo] != self.ids[lo:hi]:
            raise ValueError('Tokenizer labels differ from the trained decision head')
        self.all_ids = list(dict.fromkeys(self.ids + self.label_ids))

    @classmethod
    def from_directory(cls, directory, encode):
        root = Path(directory)
        return cls(json.loads((root/'adapter_vllm/decision_head.json').read_text()),
                   json.loads((root/'calibration.json').read_text()), encode)

    def question(self, kind, question, options=None):
        if not isinstance(kind,str) or kind not in self.ranges or not isinstance(question, str) or not question:
            raise ValueError('Provide kind noul/score/choice and a non-empty question')
        lo, hi = self.ranges[kind]
        if kind == 'choice':
            if (not isinstance(options, list) or not 2 <= len(options) <= len(self.labels)
                    or not all(isinstance(o, str) and o for o in options)):
                raise ValueError(f'Choice requires 2..{len(self.labels)} non-empty string options')
            ids = self.label_ids[:len(options)]
            bias = self.bias[lo:hi] + [0.0] * max(0, len(options)-(hi-lo))
            lines = [f'{self.labels[i]}) {o}' for i,o in enumerate(options)]
            bias = bias[:len(options)]
        else:
            default = ['false','true'] if kind == 'noul' else [str(i) for i in range(6)]
            options = default if options is None else options
            if not isinstance(options,list) or len(options) != hi-lo or not all(isinstance(o,str) for o in options):
                raise ValueError(f'{kind} requires exactly {hi-lo} options')
            ids, bias, lines = self.ids[lo:hi], self.bias[lo:hi], options
        return {'kind':kind, 'question':question, 'options':options, 'ids':ids, 'bias':bias,
                'suffix':'\n[question] '+question+'\n[options]\n'+'\n'.join(lines)+'\n[decision]:'}

    def probabilities(self, raw, prepared):
        if len(raw) != len(prepared['ids']) or not all(math.isfinite(z) for z in raw):
            raise ValueError('Invalid/non-finite decision logits')
        z = [(a+b)/self.temperatures[prepared['kind']] for a,b in zip(raw,prepared['bias'])]
        m = max(z)
        e = [math.exp(v-m) for v in z]
        total = sum(e)
        return [v/total for v in e]


def package_decision_weights(source, output, tokenizer):
    """Keep adapter/calibration and exact source head rows in the EXL3 pack.

    Only ~264 source rows are needed even for 256 options. Slicing the source
    shard keeps the full BF16 vocabulary head out of host working memory.
    """
    root, dest = Path(source), Path(output)
    if not (root/'adapter_vllm/decision_head.json').is_file():
        return
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    profile = DecisionProfile.from_directory(root, lambda s: tokenizer.encode(s).flatten().tolist())
    index = json.loads((root/'model.safetensors.index.json').read_text())
    head_file = root/index['weight_map']['lm_head.weight']
    with safe_open(head_file, framework='pt', device='cpu') as f:
        rows = torch.cat([f.get_slice('lm_head.weight')[i:i+1] for i in profile.all_ids])
    save_file({'token_ids':torch.tensor(profile.all_ids), 'base_rows':rows.contiguous()},
              dest/'decision_rows.safetensors')
    os.chmod(dest/'decision_rows.safetensors',0o644)
    shutil.copytree(root/'adapter_vllm',dest/'adapter_vllm',dirs_exist_ok=True)
    shutil.copy2(root/'calibration.json',dest/'calibration.json')
    (dest/'decision_config.json').write_text(json.dumps({
        'format':'jev-bare-v1', 'base_rows':'decision_rows.safetensors', 'adapter':'adapter_vllm',
        'calibration':'calibration.json', 'head_precision':'source dtype; FP32 readout',
        'source_revision':(root/'source_revision.txt').read_text().strip()
            if (root/'source_revision.txt').exists() else None},indent=2)+'\n')
