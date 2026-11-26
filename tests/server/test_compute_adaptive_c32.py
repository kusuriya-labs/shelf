import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import ComputeHarness, prompt_ids


def main():
    draft = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
    if not draft:
        print("SKIP: SHELF_DFLASH2_MODEL_DIR is required")
        return 77
    prompts = [prompt_ids(f"List the integers from {100+i} to {200+i} in order, separated by commas. Include every integer.")
               for i in range(32)]
    with ComputeHarness(max_concurrent_requests=32, max_num_batched_tokens=256, max_seq_len=256,
                        auto_arena=True, dflash2_model_dir=draft, dflash2_drafts=5,
                        adaptive_dflash=True) as harness:
        harness.wait_ready()
        reported = {}
        for batch in range(6):
            streamed = {}
            pending = set()
            for i, prompt in enumerate(prompts):
                rid = batch * 32 + i + 1
                pending.add(rid)
                streamed[rid] = []
                harness.send(dict(op="generate", request_id=rid, input_ids=prompt, max_new_tokens=48,
                                  temperature=0.0, ignore_eos=True))
            while pending:
                event = harness.recv(timeout=180)
                rid = event.get("request_id")
                if event.get("event") == "error":
                    raise RuntimeError(str(event))
                if event.get("event") == "token":
                    streamed[rid].append(event["token_id"])
                if event.get("event") == "done":
                    assert rid in pending and event["finish_reason"] == "length", event
                    assert len(streamed[rid]) == 48 and streamed[rid] == event["generated_ids"]
                    assert all(token >= 0 for token in streamed[rid])
                    reported[rid] = event["speculative_stats"]
                    pending.remove(rid)
    sessions = [dict(field.split("=", 1) for field in line.split()[1:])
                for line in harness.stderr_lines if line.startswith("DFLASH2_ADAPTIVE_SESSION ")]
    assert sorted(int(s["request_id"]) for s in sessions) == list(range(1, 193))
    for session in sessions:
        assert 2 <= int(session["final_k"]) <= 5, session
        assert int(session["samples"]) > 0 and 0 <= float(session["acc_ema"]) <= 1, session
        counters = reported[int(session["request_id"])]
        assert counters["final_k"] == int(session["final_k"])
        assert counters["k_switches"] == int(session["switches"])
        assert counters["adaptive_samples"] == int(session["samples"])
        assert abs(counters["acc_ema"] - float(session["acc_ema"])) < 1e-6
        assert counters["verify_rows"] >= counters["proposed_drafts"] >= counters["accepted_drafts"]
        assert int(session["switches"]) >= 0, session
    stats = dict(line.strip().split("=", 1) for line in harness.stderr_lines
                 if line.startswith(("DFLASH2_FULL_ACCEPT_RATE=", "DFLASH2_ACCEPTANCE_RATE=", "DFLASH2_PROPOSED_DRAFTS=")))
    assert int(stats["DFLASH2_PROPOSED_DRAFTS"]) > 0
    assert 0 <= float(stats["DFLASH2_ACCEPTANCE_RATE"]) <= 1
    assert 0 <= float(stats["DFLASH2_FULL_ACCEPT_RATE"]) <= 1
    print("同時32要求で6回、ストリームと確定した生成列が一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
