import math
import random
import unittest


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sequential(parent, q, k, v, beta, gate, initial):
    states, outputs = [], []
    for i, p in enumerate(parent):
        state = initial if p < 0 else states[p]
        decay = math.exp(gate[i])
        delta = [beta[i] * (v[i][h] - decay * dot(k[i], [row[h] for row in state]))
                 for h in range(len(v[i]))]
        updated = [[decay * state[d][h] + k[i][d] * delta[h]
                    for h in range(len(delta))] for d in range(len(k[i]))]
        states.append(updated)
        outputs.append([dot(q[i], [row[h] for row in updated]) for h in range(len(delta))])
    return states, outputs


def tree_chunk(parent, q, k, v, beta, gate, initial):
    paths, cumulative, deltas, outputs = [], [], [], []
    for i, p in enumerate(parent):
        path = [] if p < 0 else paths[p].copy()
        paths.append(path + [i])
        cumulative.append(gate[i] + (0.0 if p < 0 else cumulative[p]))
        delta = []
        for h in range(len(v[i])):
            rhs = beta[i] * (v[i][h] - math.exp(cumulative[i]) *
                             dot(k[i], [row[h] for row in initial]))
            rhs -= sum(beta[i] * dot(k[i], k[j]) *
                       math.exp(cumulative[i] - cumulative[j]) * deltas[j][h]
                       for j in path)
            delta.append(rhs)
        deltas.append(delta)
        outputs.append([math.exp(cumulative[i]) * dot(q[i], [row[h] for row in initial]) +
                        sum(math.exp(cumulative[i] - cumulative[j]) * dot(q[i], k[j]) *
                            deltas[j][h] for j in paths[i]) for h in range(len(delta))])
    return paths, deltas, outputs


class TreeChunkReferenceTest(unittest.TestCase):
    def test_random_trees_and_replay(self):
        rng = random.Random(20261111)
        max_error = 0.0
        for n in (1, 8, 10, 16, 32):
            for shape in ('chain', 'star', 'mixed'):
                for _ in range(20):
                    parent = [-1] + [i - 1 if shape == 'chain' else
                                     0 if shape == 'star' else rng.randrange(i)
                                     for i in range(1, n)]
                    vec = lambda m: [rng.uniform(-0.3, 0.3) for _ in range(m)]
                    q, k, v = [vec(7) for _ in parent], [vec(7) for _ in parent], [vec(5) for _ in parent]
                    beta = [rng.uniform(0.05, 0.95) for _ in parent]
                    gate = [rng.uniform(-0.5, 0.0) for _ in parent]
                    initial = [vec(5) for _ in range(7)]
                    states, expected = sequential(parent, q, k, v, beta, gate, initial)
                    paths, delta, actual = tree_chunk(parent, q, k, v, beta, gate, initial)
                    for i in range(n):
                        for a, b in zip(actual[i], expected[i]):
                            max_error = max(max_error, abs(a - b))
                            self.assertAlmostEqual(a, b, delta=1e-5)
                        replay = [row.copy() for row in initial]
                        for j in paths[i]:
                            replay = [[math.exp(gate[j]) * replay[d][h] + k[j][d] * delta[j][h]
                                       for h in range(5)] for d in range(7)]
                        for a, b in zip(replay, states[i]):
                            for x, y in zip(a, b):
                                max_error = max(max_error, abs(x - y))
                                self.assertAlmostEqual(x, y, delta=1e-5)
        print(f'max absolute error: {max_error:.3e}')


if __name__ == '__main__':
    unittest.main()
