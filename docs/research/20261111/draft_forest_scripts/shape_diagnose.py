#!/usr/bin/env python3
"""Break down recorded chain-context tree misses; no GPU is required.

The target sequence is valid only while its prefix matches the recorded chain.
Categories partition the *first* missing edge, not the oracle token gap. Rows
on a different parent path are counted separately and overlap those categories.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from shape_upper_bound import best_first, walk


def diagnose(record, budget):
    target, generated, rounds = record['target'], record['generated'], record['rounds']
    origin = rounds[0][0]
    causes, ranks, details = Counter(), Counter(), Counter()
    eligible = oracle_gap = wasted = wrong_parent_rows = selected = 0
    for pos, anchor, drafts, choices in rounds:
        index = pos - origin
        if index < 0 or index >= len(generated) or generated[index] != anchor:
            raise ValueError(f'anchor mismatch at {pos}')
        if generated[:index + 1] != target[:index + 1]:
            continue
        truth = target[index + 1:index + 1 + len(choices)]
        nodes = best_first(choices, budget)
        current = walk(nodes, truth)
        oracle = 0
        for depth, (row, token) in enumerate(zip(choices, truth)):
            if oracle == budget - 1 or (token not in [t for t, _ in row] and token != drafts[depth]):
                break
            oracle += 1
        eligible += 1
        oracle_gap += oracle - current
        selected += len(nodes) - 1
        # A row is useful only if its entire ancestry agrees with the target.
        wasted += sum(1 for i, (parent, token, depth) in enumerate(nodes) if i and (
            depth > len(truth) or token != truth[depth - 1] or
            any(nodes[j][1] != truth[nodes[j][2] - 1] for j in ancestors(nodes, parent))))
        # Exclude root children: their parent is the matched anchor. A parent
        # beyond the recorded truth suffix is also not known to match.
        for parent, _, _ in nodes[1:]:
            if parent and (nodes[parent][2] > len(truth) or any(
                    nodes[j][1] != truth[nodes[j][2] - 1]
                    for j in ancestors(nodes, parent))):
                wrong_parent_rows += 1
        if current >= oracle:
            continue
        depth = current + 1
        token = truth[current]
        rank = next((i + 1 for i, (t, _) in enumerate(choices[current]) if t == token), None)
        # A chain-only token is in the oracle's relaxed candidate set, but not
        # in the marginal-only selector's candidate set.
        if rank is None:
            cause = 'chain_only'
        elif max(node[2] for node in nodes) < depth:
            cause = 'depth_cutoff'
        elif any(d == depth and t == token for _, t, d in nodes):
            cause = 'wrong_parent'
        else:
            cause = 'not_selected'
        parent_token = anchor if current == 0 else truth[current - 1]
        causes[cause] += 1
        ranks[str(rank) if rank else 'chain-only'] += 1
        details[f'depth={depth} rank={rank or "chain-only"} parent={parent_token} cause={cause}'] += 1
    return {'eligible_rounds': eligible, 'oracle_gap_tokens': oracle_gap,
            'first_miss': dict(causes), 'first_miss_rank': dict(ranks),
            'selected_rows': selected, 'wrong_path_rows': wasted,
            'wrong_parent_rows': wrong_parent_rows,
            'details': dict(sorted(details.items()))}


def ancestors(nodes, row):
    while row > 0:
        yield row
        row = nodes[row][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('record_dir', type=Path)
    parser.add_argument('--budgets', type=int, nargs='+', default=[10, 12, 16])
    args = parser.parse_args()
    for path in sorted(args.record_dir.glob('*.json')):
        record = json.loads(path.read_text())
        for budget in args.budgets:
            print(json.dumps({'prompt': path.stem, 'budget': budget, **diagnose(record, budget)}, sort_keys=True))


if __name__ == '__main__':
    main()
