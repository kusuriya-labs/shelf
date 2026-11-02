#!/usr/bin/env python3
"""Independent fp64 oracle tokens for Qwen3.5 greedy inference.

Wraps the external qwen35-f64-oracle binary (NOT part of this repository; see
docs/developer/testing_f64_oracle.md). The oracle is a dependency-free CPU f64
implementation of the model family and serves as a neutral referee: it
adjudicates token-level disagreements between GPU implementations and any
"correctness" path, so a broken reference cannot masquerade as truth.

usage:
  tools/qwen35_oracle_tokens.py --model-dir models/Qwen3.5-4B \
      --ids "248045,846" [--chat] [--max-new-tokens 4] [--dtype f64]

  tools/qwen35_oracle_tokens.py --model-dir models/Qwen3.5-4B \
      --prompt "What is my favorite color?" --chat --max-new-tokens 2

prints: ORACLE_FIRST_TOKENS=<ids> (parse like the bench's GREEDY_FIRST_TOKENS)
"""

import argparse
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path


def _vocab_size(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return int(cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size"))


def _encode(model_dir, prompt, chat):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    if chat:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True)
    else:
        ids = tok.encode(prompt)
    return list(ids)


def _read_last_row(logits_path, vocab):
    data = logits_path.read_bytes()
    if len(data) % (vocab * 8) != 0:
        raise SystemExit(f"bad logits size {data[len(data)-1:]}")
    rows = len(data) // (vocab * 8)
    off = (rows - 1) * vocab * 8
    return list(struct.unpack_from(f"<{vocab}d", data, off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--dtype", default="f64", choices=["f64", "bf16", "f16"])
    ap.add_argument("--attn", default=None, choices=["eager", "flash"])
    ap.add_argument("--gdn", default=None, choices=["chunked", "recurrent"])
    ap.add_argument("--oracle-dir", default="/root/qwen35-f64-oracle")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--compare-ids", default=None,
                    help="comma-separated ids to compare against the oracle tokens")
    args = ap.parse_args()

    oracle_bin = Path(args.oracle_dir) / "qwen35-oracle"
    if not oracle_bin.exists():
        raise SystemExit(
            f"oracle binary not found: {oracle_bin}\n"
            "clone https://github.com/mjsabby/qwen35-f64-oracle and run make "
            "(kept outside this repository for license reasons; see "
            "docs/developer/testing_f64_oracle.md)")

    if args.ids:
        ids = [int(x) for x in args.ids.replace(",", " ").split()]
    elif args.ids_file:
        ids = [int(x) for x in Path(args.ids_file).read_text().split()]
    elif args.prompt:
        ids = _encode(args.model_dir, args.prompt, args.chat)
    else:
        ap.error("one of --ids/--ids-file/--prompt is required")

    vocab = _vocab_size(args.model_dir)
    out_dir = Path(tempfile.mkdtemp(prefix="qwen35_oracle_"))
    view = list(ids)
    generated = []
    for _ in range(args.max_new_tokens):
        cmd = [str(oracle_bin),
               "--model", args.model_dir,
               "--ids", " ".join(str(t) for t in view),
               "--dtype", args.dtype,
               "--out", str(out_dir),
               "--threads", str(args.threads)]
        if args.attn:
            cmd += ["--attn", args.attn]
        if args.gdn:
            cmd += ["--gdn", args.gdn]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            sys.stderr.write(run.stdout[-2000:] + run.stderr[-2000:])
            raise SystemExit(f"oracle failed rc={run.returncode}")
        logits = _last_row(out_dir / "logits.bin", vocab)
        next_tok = max(range(vocab), key=logits.__getitem__)
        generated.append(next_tok)
        view.append(next_tok)
        shutil.rmtree(out_dir / "logits.bin", ignore_errors=True)

    line = ",".join(str(t) for t in generated)
    print(f"ORACLE_FIRST_TOKENS={line} context={len(ids)}")

    if args.compare_ids:
        want = [int(x) for x in args.compare_ids.split(",")]
        ok = generated == want
        print(("OK tokens match" if ok else
               f"MISMATCH got [{line}] want [{','.join(map(str, want))}]"))
        if not ok:
            raise SystemExit(1)

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_dir)
        print("ORACLE_DECODED=" + repr(tok.decode(generated)))
        print("ORACLE_DECODED_CONTEXT=" + repr(tok.decode(view)))
    except Exception:
        pass


def _last_row(logits_path, vocab):
    data = logits_path.read_bytes()
    rows = len(data) // (vocab * 8)
    if rows * vocab * 8 != len(data):
        raise SystemExit(f"bad logits.bin size {len(data)}")
    off = (rows - 1) * vocab * 8
    return list(struct.unpack_from(f"<{vocab}d", data, off))


if __name__ == "__main__":
    main()
