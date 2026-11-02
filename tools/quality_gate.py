#!/usr/bin/env python3
"""FP8 KV quality gate: perplexity / KL / top-1 / token agreement / fixed end-task."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BETTERBENCH_CORPUS = REPO_ROOT / "vendor" / "BetterBench" / "corpus" / "v1"
CORPUS_CATEGORIES = (
    "chat",
    "code",
    "file_edit",
    "json",
    "math",
    "prose",
    "reasoning",
    "summarization",
)
DEFAULT_CASES = REPO_ROOT / "tests" / "data" / "quality" / "reasoning.json"
GSM8K_CASES = REPO_ROOT / "tests" / "data" / "quality" / "gsm8k.json"
GROUPED_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_kv_output(text: str) -> dict:
    values: dict[str, object] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if not key or not key.replace("_", "").isalnum():
            continue
        try:
            values[key] = float(value)
        except ValueError:
            values[key] = value
    return values


def normalize_text(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace("$", " ")
    return re.sub(r"\s+", " ", text).strip()


def answer_matches(text: str, expected: str) -> bool:
    normalized = normalize_text(text)
    if re.fullmatch(r"-?\d+(?:\.\d+)?", expected):
        target = float(expected)
        for match in re.finditer(r"-?\d+(?:\.\d+)?", normalized):
            if float(match.group()) == target:
                return True
        return False
    return expected.lower() in normalized.lower()


def numeric_value(text: str) -> float | None:
    normalized = text.replace(",", "")
    if PLAIN_NUMBER.fullmatch(normalized):
        return float(normalized)
    return None


def extract_gsm8k_answer(text: str) -> str | None:
    marker = text.rfind("####")
    if marker >= 0:
        tokens = GROUPED_NUMBER.findall(text[marker + 4:])
        return tokens[0].replace(",", "") if tokens else None
    tokens = GROUPED_NUMBER.findall(text)
    return tokens[-1].replace(",", "") if tokens else None


def gsm8k_answer_matches(text: str, expected: str, extracted: str | None) -> bool:
    target = numeric_value(expected)
    if target is None:
        return expected.casefold() in normalize_text(text).casefold()
    if extracted is None:
        return False
    value = numeric_value(extracted)
    return value is not None and value == target


def token_agreement(reference: list[int], candidate: list[int]) -> dict:
    shared = min(len(reference), len(candidate))
    prefix = 0
    for index in range(shared):
        if reference[index] != candidate[index]:
            break
        prefix += 1
    overlap = sum(1 for index in range(shared) if reference[index] == candidate[index])
    return {
        "tokens": shared,
        "prefix_match": prefix,
        "prefix_ratio": prefix / shared if shared else 0.0,
        "positional_match": overlap,
        "positional_ratio": overlap / shared if shared else 0.0,
        "exact": reference == candidate,
        "reference_tokens": len(reference),
        "candidate_tokens": len(candidate),
    }


class ServeSession:
    def __init__(self, argv: list[str]):
        self.next_id = 1
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._request({"op": "ping"}, expect="pong")

    def _send(self, payload: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def _read_event(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("compute serve-stdio closed stdout")
        return json.loads(line)

    def _request(self, payload: dict, expect: str) -> dict:
        self._send(payload)
        while True:
            event = self._read_event()
            if event.get("event") == expect:
                return event

    def generate(self, input_ids: list[int], max_new_tokens: int) -> list[int]:
        request_id = self.next_id
        self.next_id += 1
        self._send({
            "op": "generate",
            "request_id": request_id,
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        })
        generated: list[int] = []
        while True:
            event = self._read_event()
            if event.get("request_id") != request_id:
                continue
            kind = event.get("event")
            if kind == "token":
                generated.append(int(event["token_id"]))
            elif kind == "error":
                raise RuntimeError(f"compute error: {event.get('message')}")
            elif kind == "done":
                ids = event.get("generated_ids")
                if isinstance(ids, list):
                    generated = [int(token) for token in ids]
                return generated

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self._send({"op": "shutdown"})
            assert self.proc.stdout is not None
            while True:
                line = self.proc.stdout.readline()
                if line == "":
                    break
                event = json.loads(line)
                if event.get("event") == "shutdown":
                    break
        except (BrokenPipeError, RuntimeError, json.JSONDecodeError):
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def extract_ids(encoded) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def chat_prompt_ids(tokenizer, prompt: str) -> list[int]:
    return extract_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))


def read_bench_cases(corpus_dir: Path = BETTERBENCH_CORPUS) -> list[dict]:
    cases: list[dict] = []
    for category in CORPUS_CATEGORIES:
        path = corpus_dir / f"{category}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["category"] = category
            cases.append(record)
    return cases


def build_corpus_ids(tokenizer, cases: list[dict]) -> list[int]:
    separator = tokenizer.encode("\n", add_special_tokens=False)
    if not separator:
        separator = [198]
    ids: list[int] = []
    for case in cases:
        ids.extend(extract_ids(tokenizer.apply_chat_template(
            case["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )))
        ids.extend(int(token) for token in separator)
    return ids


def write_psktok(path: Path, ids: list[int]) -> None:
    payload = b"PSKLDTOK" + struct.pack("<IQ", 1, len(ids))
    payload += struct.pack(f"<{len(ids)}i", *ids)
    path.write_bytes(payload)


def corpus_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_argv(build_dir: Path, model_dir: str, device: int, max_seq_len: int,
                 kv_dtype: str) -> list[str]:
    return [
        str(build_dir / "phaseshift-compute"),
        "--model-dir", model_dir,
        "--serve-stdio",
        "--max-seq-len", str(max_seq_len),
        "--arena-gib", "auto",
        "--device", str(device),
        "--kv-cache-dtype", kv_dtype,
        "--prefix-cache", "0",
        "--max-concurrent-requests", "1",
        "--temperature", "0",
    ]


def run_generation_suite(build_dir: Path, model_dir: str, device: int,
                         cases: list[dict], max_seq_len: int, kv_dtype: str,
                         tokenizer) -> dict:
    session = ServeSession(compute_argv(build_dir, model_dir, device, max_seq_len, kv_dtype))
    try:
        results = []
        for case in cases:
            input_ids = chat_prompt_ids(tokenizer, case["prompt"])
            started = time.monotonic()
            generated = session.generate(input_ids, int(case.get("max_new_tokens", 128)))
            elapsed = time.monotonic() - started
            text = tokenizer.decode(generated, skip_special_tokens=True)
            gsm8k = case.get("answer_format") == "gsm8k"
            extracted = extract_gsm8k_answer(text) if gsm8k else None
            if gsm8k:
                correct = gsm8k_answer_matches(text, case["expected"], extracted)
            else:
                correct = answer_matches(text, case["expected"])
            results.append({
                "id": case["id"],
                "category": case.get("category", "reasoning"),
                "answer_format": case.get("answer_format"),
                "expected": case["expected"],
                "generated_ids": generated,
                "text": text,
                "extracted": extracted,
                "correct": correct,
                "prompt_tokens": len(input_ids),
                "seconds": elapsed,
            })
        return {"dtype": kv_dtype, "cases": results}
    finally:
        session.close()


def run_quality_dump(build_dir: Path, model_dir: str, device: int, corpus: Path,
                     context: int, positions: int, kv_dtype: str,
                     dump: Path, compare: Path | None) -> dict:
    argv = [
        str(build_dir / "phaseshift-bench"),
        "quality",
        "--model-dir", model_dir,
        "--tokens-file", str(corpus),
        "--context", str(context),
        "--positions", str(positions),
        "--prefill-chunk", str(max(256, context)),
        "--arena-gib", "auto",
        "--device", str(device),
        "--kv-cache-dtype", kv_dtype,
        "--dump", str(dump),
    ]
    if compare is not None:
        argv += ["--compare", str(compare)]
    result = subprocess.run(argv, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"phaseshift-bench quality failed (exit {result.returncode})")
    return parse_kv_output(result.stdout)


def gate_metrics(parser: argparse.Namespace, fp8: dict,
                 bf16_gen: dict, fp8_gen: dict) -> tuple[bool, list[str], dict]:
    failures: list[str] = []
    ppl_delta = float(fp8["QUALITY_PPL_DELTA_PCT"])
    kl = float(fp8["QUALITY_KL_MEAN"])
    top1 = float(fp8["QUALITY_TOP1_AGREEMENT"])

    if ppl_delta > parser.max_ppl_regression_pct:
        failures.append(
            f"ppl regression {ppl_delta:.4f}% > {parser.max_ppl_regression_pct:.4f}%")
    if kl > parser.max_kl:
        failures.append(f"mean KL {kl:.6f} > {parser.max_kl:.6f}")
    if top1 < parser.min_top1:
        failures.append(f"top-1 agreement {top1:.6f} < {parser.min_top1:.6f}")

    agreement = []
    mean_prefix = None
    base_correct = 0
    candidate_correct = 0
    reasoning_cases = 0
    gsm8k_base_correct = 0
    gsm8k_candidate_correct = 0
    gsm8k_case_count = 0
    if bf16_gen["cases"]:
        for base_case, candidate_case in zip(bf16_gen["cases"], fp8_gen["cases"]):
            if base_case["id"] != candidate_case["id"]:
                raise RuntimeError("generation suite case order mismatch")
            agreement.append(token_agreement(
                base_case["generated_ids"], candidate_case["generated_ids"]))
            if base_case.get("answer_format") == "gsm8k":
                gsm8k_case_count += 1
                gsm8k_base_correct += int(bool(base_case["correct"]))
                gsm8k_candidate_correct += int(bool(candidate_case["correct"]))
            else:
                reasoning_cases += 1
                base_correct += int(bool(base_case["correct"]))
                candidate_correct += int(bool(candidate_case["correct"]))
        mean_prefix = sum(item["prefix_ratio"] for item in agreement) / len(agreement)
        if mean_prefix < parser.min_token_agreement:
            failures.append(
                f"generation prefix agreement {mean_prefix:.6f} "
                f"< {parser.min_token_agreement:.6f}")
        if candidate_correct < base_correct - parser.max_endtask_regression:
            failures.append(
                f"end-task correct {candidate_correct}/{reasoning_cases} "
                f"< bf16 {base_correct} - {parser.max_endtask_regression}")
        if gsm8k_candidate_correct < gsm8k_base_correct - parser.max_gsm8k_regression:
            failures.append(
                f"gsm8k correct {gsm8k_candidate_correct}/{gsm8k_case_count} "
                f"< bf16 {gsm8k_base_correct} - {parser.max_gsm8k_regression}")

    report = {
        "ppl_baseline": float(fp8["QUALITY_PPL_BASELINE"]),
        "ppl_candidate": float(fp8["QUALITY_PPL_CANDIDATE"]),
        "ppl_delta_pct": ppl_delta,
        "kl_mean": kl,
        "top1_agreement": top1,
        "top5_overlap": float(fp8["QUALITY_TOP5_OVERLAP"]),
        "logit_max_abs_delta": float(fp8["QUALITY_LOGIT_MAX_ABS_DELTA"]),
        "positions": int(fp8["QUALITY_POSITIONS"]),
        "generation_mean_prefix_agreement": mean_prefix,
        "generation": agreement,
        "endtask_baseline_correct": base_correct if bf16_gen["cases"] else None,
        "endtask_candidate_correct": candidate_correct if bf16_gen["cases"] else None,
        "endtask_cases": reasoning_cases,
        "gsm8k_baseline_correct": gsm8k_base_correct,
        "gsm8k_candidate_correct": gsm8k_candidate_correct,
        "gsm8k_cases": gsm8k_case_count,
    }
    return not failures, failures, report


def psktok_count(path: Path) -> int:
    data = path.read_bytes()[:20]
    if len(data) < 20 or data[:8] != b"PSKLDTOK":
        raise ValueError(f"not a PSKLDTOK file: {path}")
    version, count = struct.unpack_from("<IQ", data, 8)
    if version != 1:
        raise ValueError(f"unsupported PSKLDTOK version {version}: {path}")
    return int(count)


def cmd_run(parser: argparse.Namespace) -> int:
    build_dir = Path(parser.build_dir).resolve()
    work_dir = Path(parser.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    corpus = Path(parser.corpus).resolve() if parser.corpus else None
    cases_path = Path(parser.cases).resolve()
    tokenizer = None

    for binary in ("phaseshift-bench", "phaseshift-compute"):
        if not (build_dir / binary).is_file():
            print(f"ERROR: {binary} not found in {build_dir}", file=sys.stderr)
            return 2
    if not Path(parser.model_dir).is_dir():
        print(f"ERROR: model dir not found: {parser.model_dir}", file=sys.stderr)
        return 2
    if corpus is not None and not corpus.is_file():
        print(f"ERROR: corpus not found: {corpus}", file=sys.stderr)
        return 2
    if not cases_path.is_file():
        print(f"ERROR: cases not found: {cases_path}", file=sys.stderr)
        return 2

    if corpus is None:
        tokenizer = load_tokenizer(parser.model_dir)
        bench_cases = read_bench_cases()
        ids = build_corpus_ids(tokenizer, bench_cases)
        corpus = work_dir / "corpus.psktok"
        write_psktok(corpus, ids)
        print(f"quality gate: corpus from BetterBench v1 "
              f"({len(bench_cases)} cases, {len(ids)} tokens)", flush=True)
    corpus_info = {
        "path": str(corpus),
        "tokens": psktok_count(corpus),
        "sha256": corpus_sha256(corpus),
        "source": "vendor/BetterBench/corpus/v1" if parser.corpus is None else "explicit",
    }
    print(f"  corpus tokens={corpus_info['tokens']} sha256={corpus_info['sha256'][:16]}",
          flush=True)

    bf16_dump = work_dir / "bf16.psq"
    fp8_dump = work_dir / "fp8.psq"

    print("quality gate: teacher-forced BF16 KV baseline", flush=True)
    run_quality_dump(build_dir, parser.model_dir, parser.device, corpus,
                     parser.context, parser.positions, "bf16", bf16_dump, None)
    print("quality gate: teacher-forced FP8 KV candidate", flush=True)
    fp8 = run_quality_dump(build_dir, parser.model_dir, parser.device, corpus,
                           parser.context, parser.positions, "fp8", fp8_dump, bf16_dump)
    print(f"  ppl bf16={float(fp8['QUALITY_PPL_BASELINE']):.6f} "
          f"fp8={float(fp8['QUALITY_PPL_CANDIDATE']):.6f} "
          f"delta={float(fp8['QUALITY_PPL_DELTA_PCT']):.4f}%", flush=True)
    print(f"  KL={float(fp8['QUALITY_KL_MEAN']):.6f} "
          f"top1={float(fp8['QUALITY_TOP1_AGREEMENT']):.6f} "
          f"positions={int(fp8['QUALITY_POSITIONS'])}", flush=True)

    if parser.skip_endtask:
        bf16_gen = {"dtype": "bf16", "cases": []}
        fp8_gen = {"dtype": "fp8", "cases": []}
    else:
        cases = list(json.loads(cases_path.read_text())["cases"])
        gsm8k_count = 0
        if not parser.skip_gsm8k:
            gsm8k_path = Path(parser.gsm8k_cases).resolve()
            if not gsm8k_path.is_file():
                print(f"ERROR: gsm8k cases not found: {gsm8k_path}", file=sys.stderr)
                return 2
            payload = json.loads(gsm8k_path.read_text())
            gsm8k_cases = payload.get("cases", [])
            if parser.gsm8k_limit > 0:
                gsm8k_cases = gsm8k_cases[:parser.gsm8k_limit]
            gsm8k_count = len(gsm8k_cases)
            if gsm8k_count:
                instruction = payload.get("instruction")
                if not instruction:
                    print(f"ERROR: gsm8k instruction missing: {gsm8k_path}", file=sys.stderr)
                    return 2
                cases += [
                    dict(case, prompt=f"{case['prompt']}\n\n{instruction}")
                    for case in gsm8k_cases
                ]
        if tokenizer is None:
            tokenizer = load_tokenizer(parser.model_dir)
        max_seq_len = max(
            len(chat_prompt_ids(tokenizer, case["prompt"])) +
            int(case.get("max_new_tokens", 128)) + 16
            for case in cases
        )
        max_seq_len = max(max_seq_len, parser.max_seq_len)
        print(f"quality gate: greedy generation comparison ({len(cases)} cases)", flush=True)
        bf16_gen = run_generation_suite(
            build_dir, parser.model_dir, parser.device, cases, max_seq_len, "bf16", tokenizer)
        fp8_gen = run_generation_suite(
            build_dir, parser.model_dir, parser.device, cases, max_seq_len, "fp8", tokenizer)
        base_reasoning = [
            case for case in bf16_gen["cases"] if case.get("answer_format") != "gsm8k"]
        fp8_reasoning = [
            case for case in fp8_gen["cases"] if case.get("answer_format") != "gsm8k"]
        print(f"  end-task bf16={sum(c['correct'] for c in base_reasoning)}"
              f"/{len(base_reasoning)} fp8={sum(c['correct'] for c in fp8_reasoning)}"
              f"/{len(fp8_reasoning)}",
              flush=True)
        base_gsm8k = [
            case for case in bf16_gen["cases"] if case.get("answer_format") == "gsm8k"]
        if base_gsm8k:
            fp8_gsm8k = [
                case for case in fp8_gen["cases"] if case.get("answer_format") == "gsm8k"]
            print(f"  gsm8k bf16={sum(c['correct'] for c in base_gsm8k)}/{len(base_gsm8k)} "
                  f"fp8={sum(c['correct'] for c in fp8_gsm8k)}/{len(fp8_gsm8k)}",
                  flush=True)

    passed, failures, report = gate_metrics(parser, fp8, bf16_gen, fp8_gen)
    report["corpus"] = corpus_info
    report_path = Path(parser.json).resolve() if parser.json else work_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {report_path}", flush=True)

    if not parser.keep_dumps:
        for path in (bf16_dump, fp8_dump):
            path.unlink(missing_ok=True)

    if passed:
        print("QUALITY_GATE=PASS")
        return 0
    for failure in failures:
        print(f"QUALITY_GATE_FAIL: {failure}", file=sys.stderr)
    print("QUALITY_GATE=FAIL")
    return 1


def cmd_corpus(parser: argparse.Namespace) -> int:
    if not Path(parser.model_dir).is_dir():
        print(f"ERROR: model dir not found: {parser.model_dir}", file=sys.stderr)
        return 2
    tokenizer = load_tokenizer(parser.model_dir)
    cases = read_bench_cases()
    ids = build_corpus_ids(tokenizer, cases)
    out = Path(parser.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_psktok(out, ids)
    print(f"CORPUS_CASES={len(cases)}")
    print(f"CORPUS_TOKENS={len(ids)}")
    print(f"CORPUS_SHA256={corpus_sha256(out)}")
    print(f"CORPUS_PATH={out}")
    return 0


def cmd_selftest(_: argparse.Namespace) -> int:
    failures: list[str] = []
    checks = [
        (answer_matches("The answer is 29.", "29"), True, "plain"),
        (answer_matches("129", "29"), False, "prefix digit"),
        (answer_matches("29 apples", "29"), True, "suffix word"),
        (answer_matches("1,000", "1000"), True, "thousands separator"),
        (answer_matches("$ 42.0", "42"), True, "trailing decimal"),
        (answer_matches("answer: 7", "6"), False, "wrong number"),
    ]
    for actual, expected, name in checks:
        if actual != expected:
            failures.append(f"answer match {name}")
    gsm8k_checks = [
        (extract_gsm8k_answer("reasoning...\n#### 18"), "18", "marker"),
        (extract_gsm8k_answer("#### 1,234"), "1234", "grouped digits"),
        (extract_gsm8k_answer("The answer is 42."), "42", "no marker"),
        (extract_gsm8k_answer("there is no number"), None, "no number"),
        (extract_gsm8k_answer("#### -3"), "-3", "negative"),
    ]
    for actual, expected, name in gsm8k_checks:
        if actual != expected:
            failures.append(f"gsm8k extract {name}")
    if not GSM8K_CASES.is_file():
        failures.append("gsm8k fixture missing")
    else:
        payload = json.loads(GSM8K_CASES.read_text(encoding="utf-8"))
        gsm8k_cases = payload.get("cases", [])
        if len(gsm8k_cases) != 1319:
            failures.append(f"gsm8k fixture cases {len(gsm8k_cases)}")
        required = {"id", "category", "prompt", "expected", "answer_format"}
        for case in gsm8k_cases:
            if not required.issubset(case):
                failures.append(f"gsm8k fixture fields {case.get('id')}")
                break
            if re.fullmatch(r"-?\d+(\.\d+)?", case["expected"].replace(",", "")) is None:
                failures.append(f"gsm8k fixture expected {case.get('id')}")
                break
    agreement = token_agreement([1, 2, 3, 4], [1, 2, 9, 4])
    if agreement["prefix_ratio"] != 0.5:
        failures.append("token agreement prefix")
    if agreement["positional_ratio"] != 0.75:
        failures.append("token agreement positional")
    parsed = parse_kv_output("QUALITY_PPL_BASELINE=8.5\nfoo=bar\n")
    if parsed.get("QUALITY_PPL_BASELINE") != 8.5 or parsed.get("foo") != "bar":
        failures.append("kv parse")

    thresholds = argparse.Namespace(
        max_ppl_regression_pct=1.0, max_kl=0.05, min_top1=0.95,
        min_token_agreement=0.90, max_endtask_regression=1,
        max_gsm8k_regression=2)
    metrics = {
        "QUALITY_PPL_BASELINE": 8.0,
        "QUALITY_PPL_CANDIDATE": 8.01,
        "QUALITY_PPL_DELTA_PCT": 0.125,
        "QUALITY_KL_MEAN": 0.001,
        "QUALITY_TOP1_AGREEMENT": 0.99,
        "QUALITY_TOP5_OVERLAP": 0.999,
        "QUALITY_LOGIT_MAX_ABS_DELTA": 0.5,
        "QUALITY_POSITIONS": 100.0,
    }
    baseline_cases = [{"id": "a", "generated_ids": [1, 2, 3], "correct": True}]
    candidate_cases = [{"id": "a", "generated_ids": [1, 2, 3], "correct": True}]
    passed, _, report = gate_metrics(
        thresholds, metrics, {"cases": baseline_cases}, {"cases": candidate_cases})
    if not passed:
        failures.append("gate pass case")
    if report["ppl_candidate"] != 8.01 or report["ppl_baseline"] != 8.0:
        failures.append("gate ppl fields")
    strict = argparse.Namespace(
        max_ppl_regression_pct=0.1, max_kl=0.05, min_top1=0.95,
        min_token_agreement=0.9, max_endtask_regression=1,
        max_gsm8k_regression=2)
    passed, _, _ = gate_metrics(
        strict, metrics, {"cases": baseline_cases}, {"cases": candidate_cases})
    if passed:
        failures.append("gate ppl failure case")
    passed, _, _ = gate_metrics(
        thresholds, metrics, {"cases": []}, {"cases": []})
    if not passed:
        failures.append("gate skip end-task case")
    gsm8k_baseline = [
        {"id": f"g{index}", "answer_format": "gsm8k", "generated_ids": [1, 2, 3],
         "correct": True}
        for index in range(3)
    ]
    gsm8k_regressed = [
        {"id": f"g{index}", "answer_format": "gsm8k", "generated_ids": [1, 2, 3],
         "correct": False}
        for index in range(3)
    ]
    gsm8k_tolerated = [
        {"id": "g0", "answer_format": "gsm8k", "generated_ids": [1, 2, 3], "correct": True},
        {"id": "g1", "answer_format": "gsm8k", "generated_ids": [1, 2, 3], "correct": False},
        {"id": "g2", "answer_format": "gsm8k", "generated_ids": [1, 2, 3], "correct": False},
    ]
    passed, _, report = gate_metrics(
        thresholds, metrics, {"cases": gsm8k_baseline}, {"cases": gsm8k_regressed})
    if passed:
        failures.append("gate gsm8k regression case")
    if (report["gsm8k_cases"] != 3 or report["gsm8k_baseline_correct"] != 3
            or report["gsm8k_candidate_correct"] != 0 or report["endtask_cases"] != 0):
        failures.append("gate gsm8k fields")
    passed, _, _ = gate_metrics(
        thresholds, metrics, {"cases": gsm8k_baseline}, {"cases": gsm8k_tolerated})
    if not passed:
        failures.append("gate gsm8k ok case")

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [len(messages[0]["content"])], "attention_mask": [1]}

        def encode(self, text, add_special_tokens=False):
            return [99]

    fake = FakeTokenizer()
    bench_cases = read_bench_cases()
    if len(bench_cases) == 0:
        failures.append("bench corpus empty")
    corpus_ids = build_corpus_ids(fake, bench_cases)
    if len(corpus_ids) != 2 * len(bench_cases) or corpus_ids[1] != 99:
        failures.append("bench corpus build")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corpus.psktok"
        write_psktok(path, corpus_ids)
        if psktok_count(path) != len(corpus_ids):
            failures.append("psktok count")
        if len(corpus_sha256(path)) != 64:
            failures.append("psktok sha256")

    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    if failures:
        return 1
    print("quality gate selftest ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FP8 KV quality gate")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run BF16 vs FP8 quality comparison")
    run.add_argument("--model-dir", required=True)
    run.add_argument("--build-dir", default=str(REPO_ROOT / "build"))
    run.add_argument("--work-dir", default=str(REPO_ROOT / "build" / "quality"))
    run.add_argument("--corpus", default=None,
                     help="existing PSKLDTOK file (default: build from BetterBench v1)")
    run.add_argument("--cases", default=str(DEFAULT_CASES))
    run.add_argument("--gsm8k-cases", default=str(GSM8K_CASES),
                     help="GSM8K end-task fixture (default: tests/data/quality/gsm8k.json)")
    run.add_argument("--gsm8k-limit", type=int, default=50,
                     help="leading GSM8K cases to run, 0 means all (default: 50)")
    run.add_argument("--skip-gsm8k", action="store_true",
                     help="skip the GSM8K end-task suite")
    run.add_argument("--max-gsm8k-regression", type=int, default=2,
                     help="allowed GSM8K correct-count regression against BF16 (default: 2)")
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--context", type=int, default=128)
    run.add_argument("--positions", type=int, default=0)
    run.add_argument("--max-seq-len", type=int, default=2048)
    run.add_argument("--max-ppl-regression-pct", type=float, default=1.0)
    run.add_argument("--max-kl", type=float, default=0.05)
    run.add_argument("--min-top1", type=float, default=0.95)
    run.add_argument("--min-token-agreement", type=float, default=0.90)
    run.add_argument("--max-endtask-regression", type=int, default=1)
    run.add_argument("--skip-endtask", action="store_true")
    run.add_argument("--keep-dumps", action="store_true")
    run.add_argument("--json", default=None)
    run.set_defaults(func=cmd_run)

    corpus = sub.add_parser("corpus", help="build a PSKLDTOK file from BetterBench v1")
    corpus.add_argument("--model-dir", required=True)
    corpus.add_argument("--out", required=True)
    corpus.set_defaults(func=cmd_corpus)

    selftest = sub.add_parser("selftest", help="validate fixture matching / parsing")
    selftest.set_defaults(func=cmd_selftest)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
