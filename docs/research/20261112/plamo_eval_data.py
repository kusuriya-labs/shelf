#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def check(path, expected):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"unexpected SHA-256 for {path}: {digest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tatoeba", required=True, type=Path)
    parser.add_argument("--wiki", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    check(args.tatoeba, "f80123c76d57fc02713772b043859b105b2fdcd99829b064b8328d572c617846")
    check(args.wiki, "2b8a3efac7b468cbe6432edba5f55c21e435d93873acc6727431f08d5ed328ea")
    table = pq.read_table(args.tatoeba, columns=["english", "non_english"])
    pairs = sorted(set((row["non_english"].strip(), row["english"].strip())
                       for row in table.to_pylist()
                       if isinstance(row["english"], str) and row["english"].strip()
                       and isinstance(row["non_english"], str) and row["non_english"].strip()))
    random.Random(17).shuffle(pairs)
    pairs = [{"ja": ja, "en": en} for ja, en in pairs[15000:15010]]
    if len(pairs) != 10:
        raise ValueError("held-out corpus is too short")
    extra = Path(__file__).with_name("plamo_quality_extra.jsonl")
    pairs.extend(json.loads(line) for line in extra.read_text().splitlines() if line.strip())
    with args.wiki.open("rb") as source:
        texts = ipc.open_stream(source).read_all().column("text").to_pylist()
    wiki = [text[:600] for text in texts if len(text) > 600][:4]
    if len(wiki) != 4:
        raise ValueError("WikiText corpus is too short")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("heldout.jsonl", pairs), ("wiki.jsonl", [{"text": text} for text in wiki])):
        with (args.output_dir / name).open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
