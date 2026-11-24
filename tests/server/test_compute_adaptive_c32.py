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
                    pending.remove(rid)
    print("同時32要求で6回、ストリームと確定した生成列が一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
