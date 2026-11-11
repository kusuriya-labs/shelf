#!/usr/bin/env python3
"""Exploratory online scoring on fixed chain-context recordings, not a speed benchmark."""
import argparse
from collections import Counter, defaultdict
import heapq
import json
import math
from pathlib import Path


def build(choices, anchor, budget, factors, bigrams, counts, mode):
    nodes = [(-1, anchor, 0)]
    heap = []
    def expand(parent, score):
        depth = nodes[parent][2]
        if depth >= len(choices) or (mode == 'shallow' and depth and score < math.log(0.05)):
            return
        for token, probability in choices[depth]:
            p = probability
            if mode == 'calibrated':
                p *= factors[depth]
            if mode == 'bigram':
                # Shrink the empirical conditional estimate strongly toward the marginal.
                parent_token = nodes[parent][1]
                p = (4 * p + bigrams[parent_token, token]) / (4 + counts[parent_token])
            if p > 0:
                heapq.heappush(heap, (-(score + math.log(p)), parent, token))
    expand(0, 0.)
    while heap and len(nodes) < budget:
        neg, parent, token = heapq.heappop(heap)
        nodes.append((parent, token, nodes[parent][2] + 1))
        expand(len(nodes) - 1, -neg)
    return nodes


def replay(record, budget, mode):
    target, generated, rounds = record['target'], record['generated'], record['rounds']
    origin = rounds[0][0]
    factors = [1.] * 32
    bigrams, counts = Counter(), Counter()
    total = eligible = 0
    for pos, anchor, drafts, choices in rounds:
        index = pos - origin
        if index < 0 or index >= len(generated) or generated[index] != anchor:
            raise ValueError('anchor mismatch')
        if generated[:index + 1] != target[:index + 1]:
            continue
        truth = target[index + 1:index + 1 + len(choices)]
        nodes = build(choices, anchor, budget, factors, bigrams, counts, mode)
        row = 0
        for token in truth:
            child = next((i for i, (parent, t, _) in enumerate(nodes) if i and parent == row and t == token), None)
            if child is None:
                break
            row = child
        accepted = nodes[row][2]
        total += accepted
        eligible += 1
        # Only reached target rows have valid labels for this hypothetical tree.
        for depth in range(min(accepted + 1, len(truth))):
            token = truth[depth]
            candidates = choices[depth]
            predicted = candidates[0][1] if candidates else 0.
            actual = float(bool(candidates) and candidates[0][0] == token)
            if predicted > 0:
                ratio = max(0.25, min(2., actual / predicted))
                factors[depth] = 0.9 * factors[depth] + 0.1 * ratio
            parent = anchor if depth == 0 else truth[depth - 1]
            bigrams[parent, token] += 1
            counts[parent] += 1
    return {'eligible_rounds': eligible, 'accepted': total, 'accepted_per_round': total / eligible}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('record_dir', type=Path)
    args = parser.parse_args()
    for path in sorted(args.record_dir.glob('*.json')):
        record = json.loads(path.read_text())
        for budget in (10, 12):
            for mode in ('baseline', 'calibrated', 'bigram', 'shallow'):
                print(json.dumps({'prompt': path.stem, 'budget': budget, 'mode': mode,
                                  **replay(record, budget, mode)}))


if __name__ == '__main__':
    main()
