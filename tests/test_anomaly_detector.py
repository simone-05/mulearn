import unittest

import numpy as np
from numpy import float_
from numpy.typing import NDArray

from mulearn.anomaly_detector import AnomalyDetector, SVMAnomalyDetector, IFAnomalyDetector, LOFAnomalyDetector


class BaseTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.x = np.random.rand(10, 1)
        self.y = np.random.rand(10)

    def _fit(self, anom_det, supervised=False):
        if supervised:
            ret = anom_det.fit(self.x, self.y)
        else:
            ret = anom_det.fit(self.x)
        self.assertIsInstance(ret, anom_det.__class__)

    def _test_array_method(self, anom_det, method_name, expected, dtype=float):
        method = getattr(anom_det, method_name)
        ret = method(self.x)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret.dtype, dtype)
        if dtype == float:
            np.testing.assert_allclose(ret, expected, atol=1e-5)
        else:
            np.testing.assert_array_equal(ret, expected)

class TestSVMAnomalyDetector(BaseTest):
    def setUp(self):
        super().setUp()
        self.svm_anom_det = SVMAnomalyDetector(random_state=1).fit(self.x)

    def test_unsupervised_fit(self):
        self._fit(SVMAnomalyDetector())

    def test_supervised_fit(self):
        self._fit(SVMAnomalyDetector(), supervised=True)

    def test_score_samples(self):
        expected = np.array([0.01396709, 0.11422404, 0.11422404, 0.01406838, 0.04814439, 0.06897933, 0.03586173, 0.01151234, 0.01242632, 0.03717787])
        self._test_array_method(self.svm_anom_det, 'score_samples', expected)

    def test_anomaly_score(self):
        expected = np.array([0.01396709, 0.11422404, 0.11422404, 0.01406838, 0.04814439, 0.06897933, 0.03586173, 0.01151234, 0.01242632, 0.03717787])
        self._test_array_method(self.svm_anom_det, 'anomaly_score', expected)

    def test_decision_function(self):
        expected = np.array([0.03309146, -0.06716548, -0.06716548, 0.03299017, -0.00108584, -0.02192078, 0.01119682, 0.03554621, 0.03463223, 0.00988068])
        self._test_array_method(self.svm_anom_det, 'decision_function', expected)

    def test_unsupervised_predict(self):
        expected = np.array([1, -1, -1, 1, -1, -1, 1, 1, 1, 1])
        self._test_array_method(self.svm_anom_det, 'predict', expected, int)

    def test_supervised_predict(self):
        expected = np.array([1, -1, -1, 1, 1, -1, 1, 1, 1, -1])
        self._test_array_method(SVMAnomalyDetector().fit(self.x,self.y), 'predict', expected, int)


class TestIFAnomalyDetector(BaseTest):
    def setUp(self):
        super().setUp()
        self.if_anom_det = IFAnomalyDetector(random_state=1).fit(self.x)

    def test_unsupervised_fit(self):
        self._fit(IFAnomalyDetector())

    def test_score_samples(self):
        expected = np.array([-0.42488773, -0.64141601, -0.60446862, -0.45453654, -0.44465434, -0.49027538, -0.46369096, -0.42373147, -0.41323749, -0.53145537])
        self._test_array_method(self.if_anom_det, 'score_samples', expected)

    def test_anomaly_score(self):
        expected = np.array([0.05105757, 1., 0.83807681, 0.18099447, 0.13768542, 0.33762113, 0.22111404, 0.04599023, 0., 0.51809382])
        self._test_array_method(self.if_anom_det, 'anomaly_score', expected)

    def test_decision_function(self):
        expected = np.array([0.07511227, -0.14141601, -0.10446862, 0.04546346, 0.05534566, 0.00972462, 0.03630904, 0.07626853, 0.08676251, -0.03145537])
        self._test_array_method(self.if_anom_det, 'decision_function', expected)

    def test_unsupervised_predict(self):
        expected = np.array([1, -1, -1, 1, 1, 1, 1, 1, 1, -1])
        self._test_array_method(IFAnomalyDetector().fit(self.x,self.y), 'predict', expected, int)


class TestLOFAnomalyDetector(BaseTest):
    def setUp(self):
        super().setUp()
        self.lof_anom_det = LOFAnomalyDetector(n_neighbors=5).fit(self.x)

    def test_unsupervised_fit(self):
        self._fit(LOFAnomalyDetector())

    def test_score_samples(self):
        expected = np.array([-0.96207368, -1.42713518, -1.11779684, -0.9337545, -0.99031927, -1.037625, -0.89812854, -0.9337545, -0.96207368, -1.11079017])
        self._test_array_method(self.lof_anom_det, 'score_samples', expected)

    def test_anomaly_score(self):
        expected = np.array([0.12087776, 1., 0.41524677, 0.06734502, 0.1742714, 0.26369511, 0., 0.06734502, 0.12087776, 0.40200182])
        self._test_array_method(self.lof_anom_det, 'anomaly_score', expected)

    def test_decision_function(self):
        expected = np.array([0.53792632, 0.07286482, 0.38220316, 0.5662455, 0.50968073, 0.462375, 0.60187146, 0.5662455, 0.53792632, 0.38920983])
        self._test_array_method(self.lof_anom_det, 'decision_function', expected)

    def test_unsupervised_predict(self):
        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self._test_array_method(LOFAnomalyDetector(n_neighbors=5).fit(self.x,self.y), 'predict', expected, int)


if __name__ == '__main__':
    unittest.main()
