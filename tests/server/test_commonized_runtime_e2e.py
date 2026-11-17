#!/usr/bin/env python3
"""共通 runtime の E2E 回帰テスト。

共通層へ移した DFlash2 / PrefixCache / TP / PP / RoPE / request 処理が、
モデルと設定をまたいで同じ生成列を返すことを確認する。

3 モデル × (DFlash2 有無 × Prefix 有無) をベースに、
FP8 で Rope x2 / TP1、long context を追加する。

環境変数:
    SHELF_MODEL_DIR          個別実行時のモデル（通常はケース側が上書きする）
    SHELF_DFLASH2_MODEL_DIR  DFlash2 draft モデル
    SHELF_TEST_DEVICE        TP1 のデバイス
    SHELF_TEST_TP_DEVICES    TP2 のデバイス (例: 2,3)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support  # noqa: E402
from support import Checker, ComputeHarness, prompt_ids  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODELS = {
    "fp8": REPO / "models/Qwen3.8-27B-FP8",
    "mxfp4": REPO / "models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4",
    "flash-next": REPO / "models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw",
}
DRAFTS = {
    "fp8": REPO / "models/Qwen3.8-27B-DFlash2-FP8",
    "mxfp4": REPO / "models/Qwen3.8-27B-DFlash2-FP8",
    "flash-next": REPO / "models/Qwen3.8-Flash-Next-DFlash2",
}
PROMPT = "What is 2+2? Answer with the number only."
MAX_NEW_TOKENS = 8

# (model, dflash2, prefix, rope_factor, tp, max_seq_len)
CASES = [
    ("fp8",        False, True,  "1", 2, 512),
    ("fp8",        True,  True,  "1", 2, 512),
    ("fp8",        False, False, "1", 2, 512),
    ("fp8",        True,  False, "1", 2, 512),
    ("fp8",        False, True,  "2", 2, 512),
    ("fp8",        True,  True,  "2", 2, 512),
    ("fp8",        False, True,  "1", 1, 512),
    ("fp8",        True,  True,  "1", 1, 512),
    ("mxfp4",      False, True,  "1", 2, 512),
    ("mxfp4",      True,  True,  "1", 2, 512),
    ("mxfp4",      False, False, "1", 2, 512),
    ("mxfp4",      True,  False, "1", 2, 512),
    ("flash-next", False, True,  "1", 2, 512),
    ("flash-next", True,  True,  "1", 2, 512),
    ("flash-next", False, False, "1", 2, 512),
    ("flash-next", True,  False, "1", 2, 512),
    ("mxfp4",      False, True,  "1", 2, 262144),
    ("flash-next", False, True,  "1", 2, 262144),
]


def draft_dir(model_name: str) -> Path:
    env = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
    return Path(env) if env else DRAFTS[model_name]


def devices(tp: int) -> str:
    if tp == 1:
        return os.environ.get("SHELF_TEST_DEVICE", "0")
    return os.environ.get("SHELF_TEST_TP_DEVICES", "0,1")


def paged_prompt(base, length):
    out = []
    while len(out) < length:
        out.extend(base)
    return out[:length]


def collect(harness):
    tokens = []
    while True:
        event = harness.recv()
        kind = event.get("event")
        if kind == "token":
            tokens.append(event["token_id"])
        elif kind == "done":
            return tokens, event
        elif kind == "error":
            raise RuntimeError(str(event))


def run_case(label, model_name, *, dflash2, prefix, rope, tp, max_seq_len):
    model = MODELS[model_name]
    os.environ["SHELF_MODEL_DIR"] = str(model)
    support._PROCESSOR_CACHE.clear()
    input_ids = paged_prompt(prompt_ids(PROMPT), 96)

    harness = ComputeHarness(
        model_path=model,
        auto_arena=True,
        max_seq_len=max_seq_len,
        devices=devices(tp),
        tp=tp,
        dflash2_model_dir=str(draft_dir(model_name)) if dflash2 else None,
        prefix_cache_enabled=prefix,
        rope_factor=rope,
    )
    try:
        harness.start()
        harness.wait_ready()
        results = []
        for request_id in (1, 2):
            harness.send({
                "op": "generate",
                "request_id": request_id,
                "input_ids": input_ids,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            })
            results.append(collect(harness))
        return results, harness.stderr_lines
    finally:
        harness.close()


def main() -> int:
    checker = Checker("commonized runtime e2e")
    baselines = {}

    only = os.environ.get("SHELF_E2E_ONLY")
    cases = CASES
    if only:
        indices = {int(x) for x in only.split(",")}
        cases = [case for i, case in enumerate(CASES) if i in indices]

    for model_name, dflash2, prefix, rope, tp, seq in cases:
        model = MODELS[model_name]
        draft = draft_dir(model_name)
        if not model.is_dir():
            print(f"SKIP: {model} is not a directory")
            continue
        if dflash2 and not draft.is_dir():
            print(f"SKIP: {draft} is not a directory")
            continue

        label = (f"{model_name} dflash2={int(dflash2)} prefix={int(prefix)} "
                 f"rope={rope} tp={tp} seq={seq}")
        print(f"=== {label}")

        try:
            results, stderr = run_case(
                label, model_name, dflash2=dflash2, prefix=prefix, rope=rope,
                tp=tp, max_seq_len=seq)
        except Exception as exc:  # noqa: BLE001 - E2E harness reports failures as text
            checker.check(f"{label} run", False, repr(exc))
            continue

        (tokens1, done1), (tokens2, done2) = results
        checker.check(f"{label} started", True)
        if dflash2:
            checker.check(
                f"{label} dflash2 enabled",
                any("DFLASH2_ENABLED=1" in line for line in stderr),
                str(stderr[-10:]))
        checker.check(f"{label} two runs agree", tokens1 == tokens2,
                      f"{tokens1} vs {tokens2}")
        if prefix:
            checker.check(f"{label} prefix restored",
                          done2.get("restored_tokens", 0) > 0,
                          str(done2.get("restored_tokens")))
        else:
            checker.check(f"{label} no prefix restore",
                          done1.get("restored_tokens", 0) == 0
                          and done2.get("restored_tokens", 0) == 0,
                          f"{done1.get('restored_tokens')} / {done2.get('restored_tokens')}")

        is_baseline = not dflash2 and prefix and rope == "1" and tp == 2 and seq == 512
        if is_baseline:
            baselines[model_name] = tokens1
        elif model_name in baselines:
            checker.check(f"{label} matches baseline",
                          tokens1 == baselines[model_name],
                          f"{tokens1} vs {baselines[model_name]}")

    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
