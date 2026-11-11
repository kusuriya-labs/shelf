import hashlib
import json
import struct
import sys
from pathlib import Path
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[4]
model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
sources = {
    'prose': 'vendor/BetterBench/README.md',
    'code': 'src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp',
    'reasoning': 'tests/data/quality/gsm8k.json',
    'japanese': 'docs/developer/testing.md',
}
for category, source in sources.items():
    text = (root / source).read_text()
    if category == 'reasoning':
        cases = json.loads(text)['cases']
        text = '\n\n'.join(c['prompt'] for c in cases)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < 2177:
        raise RuntimeError(f'{category}: only {len(ids)} tokens')
    ids = ids[:2177]
    payload = b'PSKLDTOK' + struct.pack('<IQ', 1, len(ids)) + struct.pack(f'<{len(ids)}i', *ids)
    (out / f'{category}.psktok').write_bytes(payload)
    print(category, len(ids), hashlib.sha256(payload).hexdigest(), source, flush=True)
