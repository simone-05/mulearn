import unittest

import numpy as np
import pickle

from mulearn import FuzzyInductor
from mulearn.anomaly_detector import IFAnomalyDetector, SVMAnomalyDetector


class TestFuzzyInductor(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.x = np.random.rand(10, 1)
        self.y = np.random.rand(10)
        self.fi = FuzzyInductor()
        self.fitted_fi = FuzzyInductor(anomaly_detector=IFAnomalyDetector(random_state=1)).fit(self.x)

    def test_serialization(self):
        fi = FuzzyInductor(anomaly_detector=IFAnomalyDetector())
        fi.fit(self.x, self.y)
        s = pickle.dumps(fi)
        fi_clone = pickle.loads(s)
        self.assertEqual(fi, fi_clone)

    def test_supervised_fit(self):
        ret1 = self.fi.fit(self.x)
        ret2 = self.fi.fit(self.x, self.y)
        self.assertIsInstance(ret1, FuzzyInductor().__class__)
        self.assertIsInstance(ret2, FuzzyInductor(anomaly_detector=SVMAnomalyDetector()).__class__)

    def test_decision_function(self):
        expected = [1., 0.22281171, 0.28947992, 0.83739873, 0.89812757, 0.65008533, 0.78481228, 1., 1., 0.48558692]
        ret = self.fitted_fi.decision_function(self.x)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret.dtype, float)
        np.testing.assert_allclose(ret, expected, atol=1e-5)

    def test_predict(self):
        expected1 = [1., 0.22281171, 0.28947992, 0.83739873, 0.89812757, 0.65008533, 0.78481228, 1., 1., 0.48558692]
        expected2 = [1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        ret1 = self.fitted_fi.predict(self.x)
        ret2 = self.fitted_fi.predict(self.x, .3)

        self.assertIsInstance(ret1, np.ndarray)
        self.assertEqual(ret1.dtype, float)
        np.testing.assert_allclose(ret1, expected1, atol=1e-5)

        self.assertIsInstance(ret2, np.ndarray)
        self.assertEqual(ret2.dtype, int)
        np.testing.assert_allclose(ret2, expected2)

        self.assertRaises(ValueError, self.fitted_fi.predict, self.x, 100)

    def test_score(self):
        expected = -0.24700461398335993
        ret = self.fitted_fi.score(self.x, self.y)
        self.assertAlmostEqual(ret, expected, places=5)

    def test_get_profile_raiseException(self):
        self.assertRaises(RuntimeError, self.fi.get_profile)

    def test_get_profile(self):
        fi = FuzzyInductor(anomaly_detector=IFAnomalyDetector(random_state=1), keep_original_data=True).fit(self.x)
        ret = fi.get_profile()
        self.assertIsInstance(ret, list)
        self.assertEqual(len(ret), 3)

        expected0 = [0.05105757, 1, 0.83807681, 0.18099447, 0.13768542]
        expected1 = [0, 0.00552764, 0.01105528, 0.01658291, 0.02211055]
        expected2 = [1, 1, 1, 1, 1]
        self.assertIsInstance(ret[0], np.ndarray)
        self.assertEqual(ret[0].dtype, float)
        np.testing.assert_allclose(ret[0][:5], expected0, atol=1e-5)

        self.assertIsInstance(ret[1], np.ndarray)
        self.assertEqual(ret[1].dtype, float)
        np.testing.assert_allclose(ret[1][:5], expected1, atol=1e-5)

        self.assertIsInstance(ret[2], np.ndarray)
        self.assertEqual(ret[2].dtype, float)
        np.testing.assert_allclose(ret[2][:5], expected2, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
