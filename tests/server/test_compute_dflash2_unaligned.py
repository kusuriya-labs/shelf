#!/usr/bin/env python3
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ComputeHarness, model_dir
from test_compute_dflash2 import collect, generated_ids, run_single_shot

PROMPT = [
    248045, 846, 198, 40, 2688, 8842, 264, 220, 18, 10902, 8151, 310,
    77916, 303, 6324, 13, 353, 1040, 3487, 321, 11107, 11, 524, 48874,
    13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271, 20740, 5576,
    1892, 77916, 303, 6324, 369, 22316, 321, 4088, 470, 13, 353, 628,
    1870, 488, 264, 3487, 9485, 12, 35265, 5516, 82, 70426, 13, 3054,
    488, 1318, 310, 2830, 264, 1834, 8151, 310, 17106, 2103, 30, 248046,
    198, 248045, 846, 198, 9175, 11, 2830, 17106, 2103, 383, 1834, 220,
    17, 13, 20052, 728, 279, 1834, 220, 17, 3019, 1132, 13, 248046,
    198, 248045, 74455, 198, 248068, 271, 248069, 271,
]


def main():
    draft = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
    if not draft or not model_dir().is_dir():
        print("SKIP: target または DFlash2 model が指定されていない")
        return 77
    checker = Checker("dflash2-unaligned-prefill")
    device = int(os.environ.get("SHELF_TEST_DEVICE", "1"))
    baseline = run_single_shot(PROMPT, 256, device=device,
                               kv_cache_dtype="bf16", dflash2=False)
    extension = PROMPT[:96] + PROMPT[10:18]
    baseline_extension = run_single_shot(extension, 256, device=device,
                                         kv_cache_dtype="bf16", dflash2=False)
    for cache_enabled in (False, True):
        harness = ComputeHarness(max_seq_len=512, device=device, auto_arena=True,
                                 dflash2_model_dir=draft,
                                 prefix_cache_enabled=cache_enabled)
        try:
            harness.start()
            harness.wait_ready()
            harness.send({"op": "generate", "request_id": 1, "input_ids": PROMPT,
                          "max_new_tokens": 256, "temperature": 0.0})
            _, done, errors = collect(harness, 1)
            label = f"prefix_cache={int(cache_enabled)}"
            checker.check(f"{label} 生成が完了", not errors and done is not None, str(errors))
            if done is not None:
                actual = generated_ids(done)
                checker.check(f"{label} 非投機 greedy と完全一致", actual == baseline,
                              f"baseline={baseline} actual={actual}")
            if cache_enabled:
                harness.send({"op": "generate", "request_id": 2, "input_ids": extension,
                              "max_new_tokens": 256, "temperature": 0.0})
                _, ext_done, ext_errors = collect(harness, 2)
                checker.check("prefix 再利用の生成が完了", not ext_errors and ext_done is not None,
                              str(ext_errors))
                if ext_done is not None:
                    checker.check("prefix 96 token を復元", ext_done.get("restored_tokens") == 96,
                                  str(ext_done.get("restored_tokens")))
                    checker.check("prefix 再利用も非投機 greedy と完全一致",
                                  generated_ids(ext_done) == baseline_extension,
                                  f"baseline={baseline_extension} actual={generated_ids(ext_done)}")
        finally:
            harness.close()
    return checker.done()


if __name__ == "__main__":
    raise SystemExit(main())
