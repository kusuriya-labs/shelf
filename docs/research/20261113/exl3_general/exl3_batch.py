import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/server'))
from support import ComputeHarness


def run(args):
    os.environ['SHELF_COMPUTE'] = str(args.compute.resolve())
    prompts = [[int(x) for x in p.read_text().split()] for p in args.prompts]
    harness = ComputeHarness(model_path=args.model, auto_arena=True, max_seq_len=512, log_path=args.log,
                             max_concurrent_requests=args.concurrency,
                             dflash2_model_dir=args.drafter if args.spec else None,
                             prefix_cache_enabled=False, batch_trace=args.trace)
    results = []
    harness.start()
    try:
        harness.wait_ready()
        for iteration in range(-1, args.runs):
            requests = []
            id_base = 1000 + (iteration + 1) * args.concurrency
            for index in range(args.concurrency):
                requests.append({'op': 'generate', 'request_id': id_base + index,
                                 'input_ids': prompts[index % len(prompts)],
                                 'max_new_tokens': args.tokens, 'temperature': 0.0,
                                 'top_p': 1.0, 'top_k': 0, 'seed': 0})
            begin = time.monotonic()
            harness.send_batch(requests)
            done = {}
            streamed = {id_base + i: [] for i in range(args.concurrency)}
            while len(done) < args.concurrency:
                line = harness.proc.stdout.readline()
                if not line:
                    raise RuntimeError('compute が stdout を閉じた')
                event = json.loads(line)
                rid = event.get('request_id')
                if event.get('event') == 'error':
                    raise RuntimeError(str(event))
                if rid in streamed and event.get('event') == 'token':
                    streamed[rid].append(event['token_id'])
                if rid in streamed and event.get('event') == 'done':
                    done[rid] = event
            elapsed = time.monotonic() - begin
            ids = [done[id_base + i]['generated_ids'] for i in range(args.concurrency)]
            if any(ids[i] != streamed[id_base + i] for i in range(args.concurrency)):
                raise RuntimeError('stream と完了時の token が不一致')
            if iteration >= 0:
                count = sum(map(len, ids))
                results.append({'iteration': iteration, 'elapsed_s': elapsed,
                                'tokens': count, 'tokens_per_s': count / elapsed,
                                'generated_ids': ids,
                                'finish_reason': [done[id_base + i]['finish_reason'] for i in range(args.concurrency)]})
        return {'concurrency': args.concurrency, 'spec': args.spec,
                'compute': str(args.compute), 'runs': results,
                'batch_steps': harness.batch_steps() if args.trace else []}
    finally:
        harness.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--drafter')
    parser.add_argument('--spec', action='store_true')
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--prompts', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    if args.spec and not args.drafter:
        parser.error('--spec には --drafter が必要')
    print(json.dumps(run(args), ensure_ascii=False), flush=True)
