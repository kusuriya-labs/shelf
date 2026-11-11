#!/usr/bin/env python3
"""Compare tree shapes on recorded rounds whose entire prefix matches a greedy target.

Run under the GPU lock; SHELF_* model/build variables must be configured. Only
rounds with an identical target prefix are eligible: a divergent future sequence
is not a target oracle for the draft's context.
"""
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'tests/server'))
from support import compute_binary, model_dir, prompt_ids
from test_compute_dflash2_tree import PROMPTS


def run(ids, tree):
    cmd = [str(compute_binary()), '--model-dir', str(model_dir()), '--input-ids-file', '/dev/stdin', '--max-new-tokens', '256', '--max-seq-len', '512', '--arena-gib', 'auto', '--device', '0', '--temperature', '0', '--top-p', '1', '--top-k', '0', '--seed', '0']
    if tree:
        cmd += ['--dflash2-model-dir', os.environ['SHELF_DFLASH2_MODEL_DIR'], '--dflash2-drafts', '7', '--dflash2-tree-budget', '10']
    env = dict(os.environ, SHELF_DFLASH2_TREE_TRACE_SHAPE='1', SHELF_DFLASH2_TREE_TRACE_CHAIN='1')
    p = subprocess.run(cmd, input='\n'.join(map(str, ids)) + '\n', text=True, capture_output=True, env=env, timeout=900)
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:])
    output = next(x for x in p.stdout.splitlines() if x.startswith('GENERATED_IDS='))
    tokens = [int(x) for x in output.split('=', 1)[1].split(',') if x.strip()]
    rounds = []
    for line in p.stderr.splitlines():
        if not line.startswith('DFLASH2_TREE_SHAPE '):
            continue
        cols = line.split(';')
        header = cols[0].split()
        _, pos, anchor, marker, *drafts = header
        if marker != 'drafts':
            raise ValueError('missing chain proposals')
        choices = [[(int(a), float(b)) for a, b in (v.split(':') for v in col.split())] for col in cols[1:]]
        rounds.append((int(pos), int(anchor), [int(x) for x in drafts], choices))
    return tokens, rounds


def best_first(choices, budget):
    nodes = [(-1, None, 0)]
    heap = []
    def expand(parent, score):
        depth = nodes[parent][2]
        if depth < len(choices):
            for token, p in choices[depth]:
                if p > 0:
                    heapq.heappush(heap, (-score - __import__('math').log(p), parent, token))
    expand(0, 0)
    while heap and len(nodes) < budget:
        negscore, parent, token = heapq.heappop(heap)
        nodes.append((parent, token, nodes[parent][2] + 1))
        expand(len(nodes) - 1, -negscore)
    return nodes


def fixed(choices, budget):
    # Top-1 spine, then top-2 leaves at the earliest positions.
    nodes = [(-1, None, 0)]
    for depth, row in enumerate(choices):
        if len(nodes) == budget or not row:
            break
        nodes.append((len(nodes) - 1, row[0][0], depth + 1))
    for depth, row in enumerate(choices):
        if len(nodes) == budget or depth >= len(nodes) or len(row) < 2:
            break
        nodes.append((depth, row[1][0], depth + 1))
    return nodes


def walk(nodes, target):
    row = 0
    for token in target:
        child = next((i for i, (parent, t, _) in enumerate(nodes) if parent == row and t == token), None)
        if child is None:
            break
        row = child
    return nodes[row][2]


def main():
    for name, prompt in zip(('shell', 'code', 'reasoning'), PROMPTS):
        ids = prompt_ids(prompt)
        target, _ = run(ids, False)
        generated, rounds = run(ids, True)
        if len(target) != 256 or len(generated) != 256 or not rounds:
            raise ValueError('incomplete generation or missing trace')
        origin = rounds[0][0]
        result = {n: {'current': 0, 'oracle': 0, 'fixed': 0} for n in (8, 10, 12, 16)}
        usable = 0
        failures = {n: {} for n in result}
        ranks = {n: {} for n in result}
        snapshot_counts = {n: [] for n in result}
        for pos, anchor, drafts, choices in rounds:
            index = pos - origin
            if index < 0 or index >= len(generated) or generated[index] != anchor:
                raise ValueError(f'anchor mismatch at {pos}: {index}')
            if generated[:index + 1] != target[:index + 1]:
                continue
            truth = target[index + 1:index + 1 + len(choices)]
            usable += 1
            for n in result:
                nodes = best_first(choices, n)
                snapshot_counts[n].append(len({parent for i, (parent, _, _) in enumerate(nodes)
                                               if i > 0 and parent >= 0 and parent != i - 1}))
                chain = [(i, token, i + 1) for i, token in enumerate(drafts)]
                current = walk([(-1, None, 0)] + chain if n == 8 else nodes, truth)
                fixed_len = walk(fixed(choices, n), truth)
                oracle = 0
                for row, token in zip(choices, truth):
                    if oracle == n - 1 or (all(t != token for t, _ in row) and token != drafts[oracle]):
                        break
                    oracle += 1
                result[n]['current'] += current
                result[n]['oracle'] += oracle
                result[n]['fixed'] += fixed_len
                if current < oracle:
                    depth = str(current + 1)
                    failures[n][depth] = failures[n].get(depth, 0) + 1
                    rank = next((str(i + 1) for i, (token, _) in enumerate(choices[current])
                                 if token == truth[current]), 'chain-only')
                    key = f'{depth}:{rank}'
                    ranks[n][key] = ranks[n].get(key, 0) + 1
        print(json.dumps({'prompt': name, 'eligible_rounds': usable, 'total_rounds': len(rounds), 'totals': result, 'miss_depths': failures, 'miss_depth_rank': ranks, 'snapshots': {n: {'sum': sum(v), 'max': max(v, default=0)} for n, v in snapshot_counts.items()}}), flush=True)

if __name__ == '__main__':
    main()
