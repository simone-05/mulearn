import unittest
import multiprocessing as mp
import numpy as np
import pickle
import itertools as it
import time
import copy
from joblib import Parallel, delayed

from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from mulearn import FuzzyInductor
from mulearn.anomaly_detector import IFAnomalyDetector, LOFAnomalyDetector, SVMAnomalyDetector
from mulearn.kernel import GaussianKernel
from mulearn.optimization import GurobiSolver


NUM_CORES = mp.cpu_count()

class TestFuzzyInductor(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.x = np.random.rand(10, 1)
        self.y = np.random.rand(10)
        self.fi = FuzzyInductor()
        self.fitted_fi = FuzzyInductor(anomaly_detector=IFAnomalyDetector(random_state=1)).fit(self.x)

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
                                                    random_state=1),
                    model_assessment=StratifiedKFold(n_splits=5,
                                                        shuffle=True,
                                                        random_state=1),
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

    def test_serialization(self):
        fis = [
            FuzzyInductor(anomaly_detector=SVMAnomalyDetector()),
            FuzzyInductor(anomaly_detector=IFAnomalyDetector(random_state=1)),
            FuzzyInductor(anomaly_detector=LOFAnomalyDetector())
        ]
        for fi in fis:
            fi.fit(self.x)
            s = pickle.dumps(fi)
            fi_clone = pickle.loads(s)
            self.assertEqual(fi, fi_clone)

    def test_train_parameters(self):
        d = load_iris()
        x = d['data']
        y = d['target']
        y[y==2] = 0
        model = FuzzyInductor(anomaly_detector=SVMAnomalyDetector(solver=GurobiSolver(time_limit=10)))
        grid = {'anomaly_detector__c': np.linspace(0.1, 0.2, 2),
                'anomaly_detector__k': [GaussianKernel(.01), GaussianKernel(.1)]}
        cv_out = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        cv_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        gs = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=cv_in, n_jobs=NUM_CORES, pre_dispatch=2*NUM_CORES, error_score='raise')
        score = cross_val_score(gs, x, y, scoring='neg_root_mean_squared_error', cv=cv_out)
        expected = np.array([-0.44679613, -0.54772247, -0.54745442, -0.54772248, -0.49874374])
        for s, e in zip(score, expected):
            self.assertAlmostEqual(s, e)

    def test_custom_train(self):
        d = load_iris()
        x = d['data']
        y = d['target']
        y[y==2] = 0
        model = FuzzyInductor(anomaly_detector=SVMAnomalyDetector(solver=GurobiSolver(time_limit=10)))
        grid = {'anomaly_detector__c': np.linspace(0.1, 0.3, 2),
                'anomaly_detector__k': [GaussianKernel(.1), GaussianKernel(.01)]}
        model, best_conf, result = self._learn_parallel(x, y, model, grid, n_jobs=NUM_CORES, pre_dispatch=2*NUM_CORES)
        result = {'configuration': best_conf} | result
        self.assertAlmostEqual(result['RMSE mean'], 0.5177007944553994, delta=1E-5)
        self.assertAlmostEqual(result['RMSE std'], 0.04491211289792065, delta=1E-4)

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
        ret2 = self.fitted_fi.predict(self.x, alpha=.3)

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
