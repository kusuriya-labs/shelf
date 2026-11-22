#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ComputeHarness, model_dir
from test_compute_dflash2 import collect, generated_ids, run_single_shot
from test_compute_dflash2_grammar_protocol import mask

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


def check_full_pages(draft, device):
    reference = {}
    for cache in (False, True):
        h = ComputeHarness(max_seq_len=256, max_concurrent_requests=1, auto_arena=True,
                           dflash2_model_dir=draft, dflash2_drafts=15, device=device,
                           prefix_cache_enabled=cache, prefix_cache_checkpoints=4)
        try:
            h.start()
            h.wait_ready()
            for length in (95, 96, 97):
                prompt = [800 + length] + [300] * (length - 1)
                h.send({'op': 'generate', 'request_id': length, 'input_ids': prompt,
                        'max_new_tokens': 17, 'temperature': 0, 'grammar': True, 'bitmask': mask(100)})
                expected = [100]
                while True:
                    event = h.recv()
                    assert event['event'] != 'error', event
                    if event['event'] == 'draft':
                        proposal = event['draft_tokens']
                        assert len(expected) == 1 and len(proposal) == 15, event
                        expected.extend(proposal + [101])
                        h.send({'op': 'draft_masks', 'request_id': length, 'round_id': event['round_id'],
                                'valid_prefix': 15, 'bitmasks': [mask(token) for token in expected[1:]]})
                    if event['event'] == 'done':
                        assert len(expected) == 17 and expected == event['generated_ids'], event
                        break
                boundary = (length + 16) // 16 * 16
                extension = (prompt + expected)[:boundary] + [191, 123]
                h.send({'op': 'generate', 'request_id': length + 100, 'input_ids': extension,
                        'max_new_tokens': 19, 'temperature': 0})
                _, done, errors = collect(h, length + 100)
                assert not errors and done is not None, errors
                assert done['restored_tokens'] == (boundary if cache else 0), done
                result = (expected, generated_ids(done))
                if cache:
                    assert result == reference[length], (length, result, reference[length])
                else:
                    reference[length] = result
        finally:
            h.close()


def main():
    draft = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
    if not draft or not model_dir().is_dir():
        print("SKIP: target または DFlash2 model が指定されていない")
        return 77
    checker = Checker("dflash2-unaligned-prefill")
    device = int(os.environ.get("SHELF_TEST_DEVICE", "1"))
    if json.loads((Path(draft) / "config.json").read_text())["dflash_config"]["block_size"] == 16:
        check_full_pages(draft, device)
        print("PASS: 15 draft の全受理と出力ページからの復元")
        return 0
    os.environ["SHELF_GDN_VERIFY_CHUNKED"] = "0"
    baseline = run_single_shot(PROMPT, 256, device=device,
                               kv_cache_dtype="bf16", dflash2=False)
    extension = PROMPT[:96] + PROMPT[10:18]
    baseline_extension = run_single_shot(extension, 256, device=device,
                                         kv_cache_dtype="bf16", dflash2=False)
    page_outputs = {}
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
            for length in (95, 96, 97):
                prompt = [1000 + length, *PROMPT[1:length]]
                for repeat in range(2):
                    request_id = length * 10 + repeat
                    harness.send({"op": "generate", "request_id": request_id, "input_ids": prompt,
                                  "max_new_tokens": 24, "temperature": 0.0})
                    _, done, errors = collect(harness, request_id)
                    case = f"{label} 入力 {length}、{repeat + 1} 回目"
                    checker.check(f"{case} の生成が完了", done is not None and not errors, str(errors))
                    if done is None:
                        continue
                    expected = (length - 1) // 16 * 16 if cache_enabled and repeat else 0
                    checker.check(f"{case} の復元位置", done["restored_tokens"] == expected,
                                  str(done["restored_tokens"]))
                    ids = generated_ids(done)
                    if not cache_enabled and repeat == 0:
                        page_outputs[length] = ids
                    checker.check(f"{case} の生成 ID が一致", page_outputs.get(length) == ids)
        finally:
            harness.close()
    return checker.done()


if __name__ == "__main__":
    raise SystemExit(main())
