import numpy as np
import unittest

from sklearn.datasets import load_iris

from mulearn.anomaly_detector import SVMAnomalyDetector, IFAnomalyDetector, LOFAnomalyDetector

#TODO - generalizza in una baseclass i test

class TestSVMAnomalyDetector(unittest.TestCase):
    def test_unsupervised_fit(self):
        ret = SVMAnomalyDetector().fit(np.random.rand(10, 1))
        self.assertIsInstance(ret, SVMAnomalyDetector)

    def test_unsupervised_predict(self):
        anom_det = SVMAnomalyDetector()
        X = np.array([[-100],[1],[2],[3],[100]])

        anom_det.fit(X)
        ret = anom_det.predict(X)

        self.assertTrue(np.array_equal(ret, np.array([-1, 1, 1, 1, -1])))

    def test_supervised_fit(self):
        ret = SVMAnomalyDetector().fit(np.random.rand(10, 1), np.random.rand(10))
        self.assertIsInstance(ret, SVMAnomalyDetector)
        
    def test_supervised_predict(self):
        anom_det = SVMAnomalyDetector()
        X = np.array([[-100],[1],[2],[3],[100]])
        y = np.array([0, 1, 1, 1, 0])

        anom_det.fit(X, y)
        ret = anom_det.predict(X)

        self.assertTrue(np.array_equal(ret, np.array([-1, 1, 1, 1, -1])))


class TestIFAnomalyDetector(unittest.TestCase):
    def test_unsupervised_fit(self):
        ret = IFAnomalyDetector().fit(np.random.rand(10, 1))
        self.assertIsInstance(ret, IFAnomalyDetector)

    def test_unsupervised_predict(self):
        anom_det = IFAnomalyDetector()
        X = np.array([[-100],[1],[2],[3],[100]])

        anom_det.fit(X)
        ret = anom_det.predict(X)

        self.assertTrue(np.array_equal(ret, np.array([-1, 1, 1, 1, -1])))


class TestLOFAnomalyDetector(unittest.TestCase):
    def test_unsupervised_fit(self):
        ret = LOFAnomalyDetector().fit(np.random.rand(10, 1))
        self.assertIsInstance(ret, LOFAnomalyDetector)

    def test_unsupervised_predict(self):
        anom_det = LOFAnomalyDetector(n_neighbors=2)
        X = np.array([[-100],[1],[2],[3],[100]])

        anom_det.fit(X)
        ret = anom_det.predict(X)

        self.assertTrue(np.array_equal(ret, np.array([-1, 1, 1, 1, -1])))


if __name__ == '__main__':
    unittest.main()
