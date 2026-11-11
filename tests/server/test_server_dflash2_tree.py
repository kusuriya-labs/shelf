#!/usr/bin/env python3
"""serve の tree verify と単一 request decoder を比較する。"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import ServerHarness, compute_binary, http_json, model_dir, prompt_ids

PROMPT = "Write a complete Python module implementing longest common subsequence, with documentation, several algorithms, validation, command-line examples, and many tests."


def serve(budget, draft, token_path):
    token_path.unlink(missing_ok=True)
    server = ServerHarness(max_seq_len=512, arena_gib=31, device=0,
                           max_concurrent_requests=1, dflash2_model_dir=draft,
                           dflash2_tree_budget=budget,
                           env={"SHELF_SERVER_TOKEN_LOG": str(token_path)},
                           startup_timeout=420)
    try:
        server.start()
        result = http_json(f"{server.base_url}/chat/completions", {
            "model": "shelf", "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0, "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        if result["choices"][0]["finish_reason"] != "length":
            raise AssertionError("generation ended before the requested length")
        return [int(x) for x in token_path.read_text().strip().split(",") if x]
    finally:
        server.close()


def single_tree(draft):
    cmd = [str(compute_binary()), "--model-dir", str(model_dir()),
           "--dflash2-model-dir", draft, "--dflash2-tree-budget", "10",
           "--input-ids-file", "/dev/stdin", "--max-new-tokens", "64",
           "--max-seq-len", "512", "--arena-gib", "auto", "--device",
           os.environ.get("SHELF_TEST_DEVICE", "0"), "--kv-cache-dtype", "bf16"]
    env = dict(os.environ, SHELF_GDN_TREE_CHUNKED="1")
    result = subprocess.run(cmd, input="\n".join(map(str, prompt_ids(PROMPT))) + "\n",
                            text=True, capture_output=True, env=env, timeout=900, check=True)
    line = next(x for x in result.stdout.splitlines() if x.startswith("GENERATED_IDS="))
    return [int(x) for x in line.split("=", 1)[1].split(",") if x.strip()]


def main():
    draft = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
    if not draft:
        raise RuntimeError("SHELF_DFLASH2_MODEL_DIR is required")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokens"
        chain = serve(0, draft, path)
        chain_tree = serve(8, draft, path)
        tree = serve(10, draft, path)
        automatic = serve(None, draft, path)
        reference = single_tree(draft)
    if len(chain) != 64 or chain_tree != chain:
        raise AssertionError("budget 8 differs from chain")
    if len(tree) != 64 or tree != reference:
        raise AssertionError("budget 10 differs from the single-request tree")
    if automatic != tree or len(automatic) != 64:
        raise AssertionError("default budget differs from budget 10 at concurrency 1")
    print("PASS: serve tree budget 8, 10 and auto")


if __name__ == "__main__":
    main()
