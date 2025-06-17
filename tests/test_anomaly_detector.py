import unittest
import logging
import warnings
import multiprocessing as mp
import itertools as it
import time
import copy

import numpy as np
from numpy.typing import NDArray
from joblib import Parallel, delayed
from sklearn.exceptions import NotFittedError
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score

from mulearn.anomaly_detector import AnomalyDetector, SVMAnomalyDetector, IFAnomalyDetector, LOFAnomalyDetector
from mulearn.kernel import LinearKernel, PolynomialKernel, GaussianKernel, HyperbolicKernel


RANDOM_STATE = 1
NUM_CORES = mp.cpu_count()

class BaseTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.x = np.random.rand(10, 1)
        self.y = np.random.rand(10)

    def _fit(self, anom_det: AnomalyDetector, supervised: bool = False):
        if supervised:
            ret = anom_det.fit(self.x, self.y)
        else:
            ret = anom_det.fit(self.x)
        self.assertIsInstance(ret, anom_det.__class__)

    def _test_array_method(self, anom_det: AnomalyDetector, method_name: str, expected: NDArray, dtype=float):
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
        self.svm_anom_det = SVMAnomalyDetector(random_state=RANDOM_STATE).fit(self.x)

    def _make_hp_configurations(self, grid):
        return [{n: v for n, v in zip(grid.keys(), t)}
                for t in it.product(*grid.values())]

    def _fit_and_score(self, estimator,
                    X_trainval, y_trainval,
                    hp_configuration, model_selection,
                    scorer=metrics.root_mean_squared_error):

        estimator.set_params(**hp_configuration)
        current_scores = []
        for train_index, val_index in model_selection.split(X_trainval, y_trainval):
            X_train, X_val = X_trainval[train_index], X_trainval[val_index]
            y_train, y_val = y_trainval[train_index], y_trainval[val_index]

            estimator.fit(X_train, y_train)
            y_hat = estimator.predict(X_val)
            score = scorer(y_val, y_hat)
            current_scores.append(score)

        return np.mean(current_scores), hp_configuration

    def _learn_parallel(self, X, y, estimator, param_grid,
                    model_selection=StratifiedKFold(n_splits=5,
                                                    shuffle=True,
                                                    random_state=RANDOM_STATE),
                    model_assessment=StratifiedKFold(n_splits=5,
                                                        shuffle=True,
                                                        random_state=RANDOM_STATE),
                    gs_scorer=metrics.root_mean_squared_error,
                    test_scorers=[metrics.root_mean_squared_error,
                                    metrics.hinge_loss],
                    test_scorer_names=['RMSE', 'Hinge'],
                    n_jobs=-1, pre_dispatch=None):

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        ping = time.time()

        outer_scores = []

        for trainval_index, test_index in model_assessment.split(X, y):
            X_trainval, X_test = X[trainval_index], X[test_index]
            y_trainval, y_test = y[trainval_index], y[test_index]

            gs_result = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)( \
                        delayed(self._fit_and_score)(copy.deepcopy(estimator),
                                            X_trainval, y_trainval,
                                            hp_conf,
                                            model_selection=model_selection,
                                            scorer=gs_scorer)
                                for hp_conf in self._make_hp_configurations(param_grid))

            best_conf = sorted(gs_result, key=lambda t: t[0])[0][1]
            estimator.set_params(**best_conf)
            estimator.fit(X_trainval, y_trainval)

            y_hat = estimator.predict(X_test)
            outer_scores.append([score(y_test, y_hat) for score in test_scorers])

        pong = time.time()
        # Refit estimator with best configuration
        # of last external cv fold on all data
        estimator.fit(X, y)

        avg = np.mean(outer_scores, axis=0)
        std = np.std(outer_scores, axis=0, ddof=1)
        result = {'model': estimator.__class__.__name__, 'type': 'FINAL'} | \
                {n + ' mean': m for n, m in zip(test_scorer_names, avg)} | \
                {n + ' std': s for n, s in zip(test_scorer_names, std)} | \
                {'time': pong-ping}

        return estimator, best_conf, result

    def test_kernels(self):
        kernel = [LinearKernel(), PolynomialKernel(2),
                  GaussianKernel(.1), HyperbolicKernel()]
        scores = [1.0529941646840044, 1.0146704466104315, 0.6980553781717808, 1.0529941646840044]
        logging.disable(logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for k, s in zip(kernel, scores):
                svm = SVMAnomalyDetector(k=k, random_state=RANDOM_STATE)
                svm.fit(self.x, self.y)
                predicted = svm.predict(self.x)
                rmse = metrics.root_mean_squared_error(self.y, predicted)
                self.assertAlmostEqual(s, rmse)
        logging.disable(logging.NOTSET)

    def test_fit_raise_notFittedError(self):
        self.assertRaises(NotFittedError, SVMAnomalyDetector().fit, np.random.rand(5, 1), warm_start=True)

    def test_unsupervised_fit(self):
        self._fit(SVMAnomalyDetector())

    def test_supervised_fit(self):
        self._fit(SVMAnomalyDetector(), supervised=True)

    #FIXME - multiprocessing problem
    # def test_train_parameters(self):
    #     x = np.random.rand(100, 1)
    #     y = np.append(np.ones(50), np.zeros(50), axis=0)
    #     model = SVMAnomalyDetector(random_state=RANDOM_STATE)

    #     grid = {'c': np.linspace(0.1, 0.2, 2),
    #             'k': [GaussianKernel(.01), GaussianKernel(.1)]}
    #     cv_out = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    #     cv_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    #     gs = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=cv_in, n_jobs=NUM_CORES, pre_dispatch=2*NUM_CORES)
    #     score = cross_val_score(gs, x, y, scoring='neg_root_mean_squared_error', cv=cv_out)
    #     expected = np.array([-0.83666003, -1.04880885, -0.70710678, -0.83666003, -0.9486833])

    #     for s, e in zip(score, expected):
    #         self.assertAlmostEqual(s, e)

    # FIXME - multiprocessing problem
    # def test_custom_train(self):
    #     x = np.random.rand(100, 1)
    #     y = np.append(np.ones(50), np.zeros(50), axis=0)
    #     model = SVMAnomalyDetector(random_state=RANDOM_STATE)

    #     grid = {'c': np.linspace(0.1, 0.3, 2),
    #             'k': [GaussianKernel(.1), GaussianKernel(.01)]}

    #     model, best_conf, result = self._learn_parallel(x, y, model, grid, n_jobs=NUM_CORES, pre_dispatch=2*NUM_CORES)

    #     result = {'configuration': best_conf} | result
    #     self.assertAlmostEqual(result['RMSE mean'], 0.8555586860711453, delta=1E-5)
    #     self.assertAlmostEqual(result['RMSE std'], 0.10012076887574035, delta=1E-4)

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
        self.if_anom_det = IFAnomalyDetector(random_state=RANDOM_STATE).fit(self.x)

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
