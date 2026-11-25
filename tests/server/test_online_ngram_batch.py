import os
from pathlib import Path

from support import ComputeHarness
from test_compute_mtp_batch import collect


PREFIX = [1, 16, 48472, 19, 22125, 16, 2225, 17, 1943, 18, 1109, 19]
SUFFIX = [16, 4160, 17, 1943, 18, 1385, 19]
SOURCES = [
    [11701, 1087, 14944, 115, 58, 352, 3157, 49020, 40, 65, 95, 48, 48, 49, 41, 96,
     44, 352, 3157, 49020, 40, 65, 95, 48, 48, 50, 41, 96, 50115, 352, 3157, 49020,
     40, 65, 95, 48, 48, 51, 41, 96, 46],
    [4882, 46],
    [3019, 58, 32, 49183, 10, 3019, 58, 32, 49183, 10, 3019, 58, 32, 49183, 10,
     3019, 58, 32, 61118, 68],
    [4111, 306, 2882, 352, 3157, 49020, 40, 65, 95, 48, 48, 52, 41, 96,
     1178, 19374, 5654, 46],
]


def run(model, count, ngram):
    min_match = 2 if count == 8 else 4
    with ComputeHarness(model_path=model, auto_arena=True, max_seq_len=256,
                        max_concurrent_requests=count, prefix_cache_enabled=False,
                        batch_trace=True, ngram_min_match=min_match if ngram else None) as harness:
        harness.wait_ready()
        requests = [
            {"op": "generate", "request_id": i, "input_ids": PREFIX + SOURCES[i % 4] + SUFFIX,
             "max_new_tokens": 128, "temperature": 0}
            for i in range(count)
        ]
        harness.send_batch(requests)
        done = collect(harness, list(range(count)))
    for event in done.values():
        assert event["finish_reason"] == "stop" and event["generated_ids"][-1] in (1, 16), event
    steps = harness.batch_steps()
    assert steps[0] == (count, sum(len(req["input_ids"]) for req in requests)), steps
    if ngram:
        assert all(rows <= 16 for _, rows in steps[1:]), steps
        if count >= 8:
            assert all(rows == n for n, rows in steps[1:] if n >= 8), steps
        assert any(n < rows for n, rows in steps[1:] if n <= min(4, count)), steps
        if count == 4:
            assert any(1 < n < rows for n, rows in steps[1:]), steps
        stats = dict(line.split("=", 1) for line in harness.stderr_lines if line.startswith("NGRAM_"))
        assert 0 < int(stats["NGRAM_HITS"]) < int(stats["NGRAM_ROUNDS"]), stats
        assert 0 < int(stats["NGRAM_ACCEPTED_DRAFTS"]) < int(stats["NGRAM_HITS"]) * 7, stats
    return [done[i]["generated_ids"] for i in range(count)]


def main():
    model = Path(os.environ.get("SHELF_PLAMO_MODEL_DIR", ""))
    if not (model / "config.json").is_file():
        print("SHELF_PLAMO_MODEL_DIR に PLaMo 3 Translate 2B BF16 を指定してください")
        return 77
    for count in (1, 4, 8):
        expected = run(model, count, False)
        actual = run(model, count, True)
        for i, (left, right) in enumerate(zip(expected, actual)):
            assert left == right, (count, i, left, right)
        if count > 1:
            assert len({len(tokens) for tokens in actual}) > 1
        print(f"c={count} 全要求のトークン列と EOS が一致", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
