import importlib.util
import unittest
import numpy as np


class TrackingTests(unittest.TestCase):
    def test_hungarian_against_exhaustive_optimum(self):
        import itertools
        from applications.tracking import linear_sum_assignment
        rng = np.random.default_rng(12)
        for n in range(1, 5):
            for m in range(1, 5):
                cost = rng.normal(size=(n, m))
                rows, cols = linear_sum_assignment(cost)
                if n <= m:
                    optimum = min(sum(cost[i, j] for i, j in enumerate(columns))
                                  for columns in itertools.permutations(range(m), n))
                else:
                    optimum = min(sum(cost[i, j] for j, i in enumerate(indices))
                                  for indices in itertools.permutations(range(n), m))
                self.assertAlmostEqual(cost[rows, cols].sum(), optimum)

    def test_translation_association_and_retention(self):
        from applications.tracking import WorkerTracker
        tracker = WorkerTracker(confirm_hits=1, max_age=2)
        first = tracker.update([[10, 10, 20, 30, .9]])[0]
        identity = first.identity
        flow = np.zeros((60, 60, 2)); flow[..., 0] = 10
        second = tracker.update([[20, 10, 30, 30, .9]], flow)[0]
        self.assertEqual(identity, second.identity)
        self.assertEqual(len(tracker.update([], flow)), 1)
        self.assertEqual(len(tracker.update([], flow)), 1)
        self.assertEqual(len(tracker.update([], flow)), 0)

    def test_mad_outlier_rejection(self):
        from applications.tracking import robust_displacement
        flow = np.ones((30, 30, 2)); flow[10, 10] = 1e6
        np.testing.assert_allclose(robust_displacement(flow, [0, 0, 30, 30]), [1, 1])
