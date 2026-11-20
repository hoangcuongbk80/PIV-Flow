import unittest
import numpy as np
from applications.conveyor import project, interval_speed, CausalSpeedWindow, reference_interval, speed_metrics
from evaluation.statistics import hierarchical_interval, paired_rows


class ConveyorTests(unittest.TestCase):
    def test_homography_translation_speed(self):
        flow = np.zeros((40, 40, 2), np.float32); flow[..., 0] = 2
        calibration = {"homography": [[.01, 0, 0], [0, .01, 0], [0, 0, 1]],
                       "belt_polygon": [[0, 0], [39, 0], [39, 39], [0, 39]], "belt_direction": [1, 0]}
        self.assertAlmostEqual(interval_speed(flow, 1 / 30, calibration), .6, places=6)
        self.assertIsNone(interval_speed(flow, 0, calibration))
        self.assertIsNone(interval_speed(flow, 1 / 30, calibration, boxes1=[[0, 0, 40, 40]]))

    def test_projective_mapping(self):
        points = np.array([[2., 3.], [10., 5.]])
        h = np.array([[1., 0, 2], [0, 1., 3], [.01, .02, 1.]])
        np.testing.assert_allclose(project(project(points, h), np.linalg.inv(h)), points)

    def test_causal_reference_independent_of_prediction(self):
        window = CausalSpeedWindow()
        for i in range(14):
            self.assertTrue(window.update(None, float(i))["warmup"])
        row = window.update(1., 14.)
        self.assertIsNone(row["prediction"])
        self.assertEqual(row["reference"], 7.)
        for _ in range(7):
            row = window.update(1., 14.)
        self.assertEqual(row["prediction"], 1.)

    def test_interval_reference_coverage(self):
        rows = [{"start_time_s": 0., "end_time_s": .5, "speed_mps": 1.},
                {"start_time_s": .5, "end_time_s": 1., "speed_mps": 3.}]
        self.assertEqual(reference_interval(rows, .25, .75), 2.)
        self.assertIsNone(reference_interval(rows, -.1, .75))
        rows[1]["start_time_s"] = .6
        self.assertIsNone(reference_interval(rows, .25, .75))

    def test_output_coverage(self):
        rows = [dict(warmup=False, prediction=.9, reference=1.), dict(warmup=False, prediction=None, reference=1.)]
        result = speed_metrics(rows)
        self.assertEqual(result["coverage_percent"], 50.)
        self.assertAlmostEqual(result["mae"], .1)

    def test_group_bootstrap_and_pairing(self):
        rows = [dict(session_id=f"s{i // 2}", recording_id=f"r{i}", delta=2.) for i in range(4)]
        result = hierarchical_interval(rows, repeats=50)
        self.assertEqual(result["ci95"], [2., 2.])
        with self.assertRaises(ValueError):
            paired_rows([dict(pair_id="a", epe=1.)], [dict(pair_id="b", epe=1.)])
