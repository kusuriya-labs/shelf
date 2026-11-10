#!/usr/bin/env python3
"""prompt が 1 chunk (2048 token) を超えるとき、DFlash2 でも prefix cache が chunk 末尾から復元され、
出力が prefix cache 無効の DFlash2 と非投機 greedy の両方に完全一致することを検証する。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ComputeHarness, model_dir
from test_compute_dflash2 import collect, generated_ids


def prompt_tokens(n, salt=0):
    return [1000 + ((i * 7919 + salt * 104729) % 20000) for i in range(n)]


def run(harness_kwargs, requests):
    harness = ComputeHarness(max_seq_len=4608, auto_arena=True, **harness_kwargs)
    results = []
    try:
        harness.start()
        harness.wait_ready()
        for request_id, ids in requests:
            harness.send({"op": "generate", "request_id": request_id, "input_ids": ids,
                          "max_new_tokens": 64, "temperature": 0.0})
            _, done, errors = collect(harness, request_id)
            results.append((done, errors))
    finally:
        harness.close()
    return results


def main():
    draft = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
    if not draft or not model_dir().is_dir():
        print("SKIP: target または DFlash2 model が指定されていない")
        return 77
    checker = Checker("dflash2-prefix-long")
    device = int(os.environ.get("SHELF_TEST_DEVICE", "1"))
    first = prompt_tokens(4300)
    # 先頭 4150 token が共通で、その後ろだけ異なる。chunk 末尾 2048 / 4096 が復元点になる。
    second = first[:4150] + prompt_tokens(30, salt=7)
    requests = [(1, first), (2, second)]

    cached = run({"device": device, "dflash2_model_dir": draft, "prefix_cache_enabled": True}, requests)
    fresh = run({"device": device, "dflash2_model_dir": draft, "prefix_cache_enabled": False}, [(2, second)])
    greedy = run({"device": device, "prefix_cache_enabled": False}, [(2, second)])

    for label, (done, errors) in (("1 回目", cached[0]), ("2 回目(cache)", cached[1]),
                                  ("cache 無効の DFlash2", fresh[0]), ("非投機", greedy[0])):
        checker.check(f"{label} の生成が完了", not errors and done is not None, str(errors))
    first_done, second_done = cached[0][0], cached[1][0]
    if first_done is not None:
        checker.check("1 回目は復元なし", first_done.get("restored_tokens", 0) == 0,
                      str(first_done.get("restored_tokens")))
    if second_done is not None:
        restored = second_done.get("restored_tokens", 0)
        checker.check("2 回目は chunk 末尾（2048 以上、page 境界）を復元",
                      restored >= 2048 and restored % 16 == 0 and restored <= 4150, str(restored))
        if fresh[0][0] is not None:
            checker.check("復元した出力が cache 無効の DFlash2 と完全一致",
                          generated_ids(second_done) == generated_ids(fresh[0][0]),
                          f"cached={generated_ids(second_done)} fresh={generated_ids(fresh[0][0])}")
        if greedy[0][0] is not None:
            checker.check("復元した出力が非投機 greedy と完全一致",
                          generated_ids(second_done) == generated_ids(greedy[0][0]),
                          f"cached={generated_ids(second_done)} greedy={generated_ids(greedy[0][0])}")
    return checker.done()


if __name__ == "__main__":
    raise SystemExit(main())
