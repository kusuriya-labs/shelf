#!/usr/bin/env python3
"""serve の tree verify と単一 request decoder を比較する。"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import ServerHarness, compute_binary, http_json, model_dir, prompt_ids

PROMPT = "Write a complete Python module implementing longest common subsequence, with documentation, several algorithms, validation, command-line examples, and many tests."


def serve(budget, draft, token_path, drafts=7):
    token_path.unlink(missing_ok=True)
    server = ServerHarness(max_seq_len=512, arena_gib=31, device=0,
                           max_concurrent_requests=1, dflash2_model_dir=draft,
                           dflash2_tree_budget=budget, dflash2_drafts=drafts,
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


def single_tree(draft, budget=10, drafts=7):
    cmd = [str(compute_binary()), "--model-dir", str(model_dir()),
           "--dflash2-model-dir", draft,
           "--dflash2-drafts", str(drafts),
           *(["--dflash2-tree-budget", str(budget)] if budget is not None else []),
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
        config = json.loads((Path(draft) / "config.json").read_text())
        if config["dflash_config"]["block_size"] == 16:
            reference = single_tree(draft, budget=0, drafts=15)
            if len(reference) != 64 or serve(0, draft, path, drafts=15) != reference:
                raise AssertionError("15 draft の server 出力が単一要求の出力と異なります")
            if serve(16, draft, path, drafts=15) != reference:
                raise AssertionError("16 行の tree budget で chain の出力が変化しました")
            print("PASS: 15 draft の compute と server の生成 ID が一致")
            return
        chain = serve(0, draft, path)
        chain_tree = serve(8, draft, path)
        tree = serve(10, draft, path)
        default_output = serve(None, draft, path)
        automatic = serve("auto", draft, path)
        reference = single_tree(draft)
        default_reference = single_tree(draft, budget=None)
        chain_reference = single_tree(draft, budget=0)
    if len(chain) != 64 or chain_tree != chain:
        raise AssertionError("budget 8 differs from chain")
    if default_reference != chain_reference or default_output != chain:
        raise AssertionError("tree budget 省略時の生成列が chain と異なります")
    if len(tree) != 64 or tree != reference:
        raise AssertionError("budget 10 differs from the single-request tree")
    if automatic != tree or len(automatic) != 64:
        raise AssertionError("同時数 1 の auto が budget 10 と異なります")
    print("PASS: serve tree budget 8, 10 and auto")


if __name__ == "__main__":
    main()
