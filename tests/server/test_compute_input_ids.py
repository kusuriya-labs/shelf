#!/usr/bin/env python3
"""Exercise text and binary token files, including non-seekable stdin, without a GPU."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile

binary = Path(os.environ.get("SHELF_BUILD_DIR", "build")) / "shelf-compute"
with tempfile.TemporaryDirectory(prefix="shelf-token-input-") as tmp:
    root = Path(tmp)
    (root / "config.json").write_text('{"model_type":"plamo3"}', encoding="utf-8")
    text = b"248041\n77091\n"
    packed = b"PSKLDTOK" + struct.pack("<IQii", 1, 2, 248041, 77091)
    (root / "tokens.txt").write_bytes(text)
    (root / "tokens.bin").write_bytes(packed)
    for name, input_data in [(str(root / "tokens.txt"), None),
                             (str(root / "tokens.bin"), None),
                             ("/dev/stdin", text), ("/dev/stdin", packed)]:
        cmd = [str(binary), "--model-dir", str(root), "--input-ids-file", name,
               "--max-new-tokens", "1", "--max-seq-len", "2", "--device", "0"]
        result = subprocess.run(cmd, input=input_data, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=10)
        if result.returncode != 2 or b"prompt + generation exceeds max_seq_len" not in result.stderr:
            raise SystemExit(f"token input failed for {name}: {result.returncode} {result.stderr!r}")
print("compute-input-ids: text and binary files and pipes passed")
