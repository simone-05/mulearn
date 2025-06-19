
"""This module implements the anomaly detectors used in mulearn.
"""

import logging
from typing import Literal, Self, Callable

import numpy as np
from numpy.typing import NDArray, ArrayLike
from numpy.random import RandomState
from sklearn.base import BaseEstimator
from sklearn.dummy import check_random_state
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import mulearn.kernel as kernel
from mulearn.optimization import Solver, GurobiSolver

logger = logging.getLogger(__name__)


class AnomalyDetector(BaseEstimator):
    """Base class for anomaly detectors.

    The base class for anomaly detectors is AnomalyDetector
    """

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs) -> Self:
        """Train the anomaly detector to be able to get the anomaly score for each data point

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Membership for the vectors in `X`, defaults to `None`.
        :type y: iterable of `float` having the same length of `X`
        :raises: RuntimeWarning if the solver found a suboptimal solution
        :raises: RuntimeError if the solver ended with an error
        :returns: self -- the trained model.
        """
        raise NotImplementedError(
            'The base class does not implement the `fit` method')

    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Is the most direct score that an implementation can give.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: array with the score for each vector of data
        """
        raise NotImplementedError(
            'The base class does not implement the `score_samples` method')

    def anomaly_score(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Get the anomaly scores for each of the original points data.
        Is used by the fuzzifier to compute the membership for each data point.
        The bigger the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- of anomaly scores, between 0 and 1.
        """
        raise NotImplementedError(
            'The base class does not implement the `anomaly_score` method')

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Shifts the `score_samples` score to be able to then predict.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- for each value: less than zero is more anomalous, more than zero is more inlier.
        """
        raise NotImplementedError(
            'The base class does not implement the `decision_function` method')

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """
        Returns the array of predictions for each data point in X.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length.
        :returns: np.array -- for each value: -1 means anomalous, +1 means normal.
        """
        raise NotImplementedError(
            'The base class does not implement the `predict` method')

    def _check_X_y(self, X: ArrayLike, y: ArrayLike | None) -> tuple[ArrayLike, ArrayLike|None]:
        """
        Function to do the checks on the arrays of X and y, mainly if they are same length and if y is not none we check that its values are between 0 and 1
        """

        X = check_array(X)

        if y is not None:
            X, y = check_X_y(X, y)
            for e in y:
                if e < 0 or e > 1:
                    raise ValueError('`y` values should belong to [0, 1]')

        return X, y


class SVMAnomalyDetector(AnomalyDetector):
    """Implementation of Support Vector Machine for AnomalyDetector.

    :param fixed_term_: fixed term of the equation
    :type fixed_term: `ArrayLike`
    :param chis_: estimated solutiton parameters
    :type chis_: `ArrayLike`
    :param squared_radius_: estimated hypersphere radius
    :type squared_radius_: `float`
    """

    def __init__(self,
                 c: float = 1,
                 k: kernel.Kernel = kernel.GaussianKernel(),
                 solver: Solver = GurobiSolver()):
        """Create an instance of :class:`SVMAnomalyDetector`.

        :param c: Trade-off constant, defaults to 1.
        :type c: `float`
        :param k: Kernel function, defaults to :class:`GaussianKernel()`.
        :type k: :class:`mulearn.kernel.Kernel`
        :param solver: Solver to be used to obtain the optimization problem
          solution, defaults to `GurobiSolver()`.
        :type solver: :class:`mulearn.optimization.Solver`
        """

        self.c = c
        self.k = k
        self.solver = solver

    def __repr__(self, **kwargs):
        return f'SVMAnomalyDetector(c={self.c}, k={self.k}, ' \
               f'solver={self.solver})'

    def __eq__(self, other):
        """Check equality w.r.t. other objects."""
        equal = (
            type(self) is type(other) and
            self.get_params() == other.get_params()
        )
        isfitted_self = hasattr(self, 'chis_')
        isfitted_other = hasattr(other, 'chis_')
        if isfitted_self and isfitted_other:
            return equal and (self.chis_ == other.chis_)
        elif isfitted_self != isfitted_other:
            return False
        return equal

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, warm_start: bool = False) -> Self:
        r"""Train the anomaly detector to be able to get the anomaly score for each data point

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the
          same length
        :param y: Membership for the vectors in `X`, defaults to `None`.
        :type y: iterable of `float` having the same length of `X`
        :param warm_start: flag triggering the non reinitialization of
          independent variables of the optimization problem, defaults to
          `None`.
        :type warm_start: `bool`
        :raises: RuntimeWarning if the solver found a suboptimal solution
        :raises: RuntimeError if the solver ended with an error
        :returns: self -- the trained model.
        """

        self._X = X.copy()

        X, y = super()._check_X_y(X, y)

        if y is None:
            y = np.ones(len(X))

        if warm_start:
            check_is_fitted(self, ['chis_'])
            if self.chis_ is None:
                raise NotFittedError('chis variable are set to None')
            self.solver.initial_values = self.chis_

        if type(self.k) is kernel.PrecomputedKernel:
            idx = X.flatten()
            self._gram = self.k.kernel_computations[idx][:, idx]
        else:
            self._gram = np.array([self.k.compute(x1, X) for x1 in X])

        self.chis_: ArrayLike = self.solver.solve(X, y, self.c, self._gram)

        self.fixed_term_ = np.array(self.chis_).dot(self._gram.dot(self.chis_))

        chi_SV_index = [i for i, (chi, mu) in enumerate(zip(self.chis_, y))
                        if -self.c * (1 - mu) < chi < self.c * mu]
        chi_sq_radius = list(self.score_samples(X[chi_SV_index]))

        if len(chi_sq_radius) == 0:
            self.chis_ = None
            logger.warning('No support vectors found')
            return self

        self.squared_radius_ = np.mean(chi_sq_radius)
        self.score_05_ = self.squared_radius_

        return self

    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        The bigger the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: array with the score for each vector of data
        """
        X = np.array(X)
        t1 = self.k.compute(X, X)
        t2 = np.array([self.k.compute(x_i, X)
                        for x_i in self._X]).transpose().dot(self.chis_)
        ret = t1 -2 * t2 + self.fixed_term_
        return ret

    def anomaly_score(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Get the anomaly scores for each of the original points data
        Is used by the fuzzifier to compute the membership for each data point.
        The bigger the more anomalous.
        For SVM is the same of score_samples.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- of anomaly scores, between 0 and 1.
        """
        return self.score_samples(X)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Shifts the `score_samples` score to be able to then predict.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- for each value: less than zero is more anomalous, more than zero is more inlier.
        """
        return self.squared_radius_ - self.score_samples(X)

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """
        Returns the array of predictions for each data point in X.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length.
        :returns: np.array -- for each value: -1 means anomalous, +1 means normal.
        """
        predictions = [1 if value>=0 else -1 for value in self.decision_function(X)]
        return np.array(predictions)


class IFAnomalyDetector(AnomalyDetector):
    def __init__(self,
                 n_trees: int = 100,
                 max_samples: Literal['auto'] | int | float = 'auto',
                 contamination: Literal['auto'] | float = 'auto',
                 max_features: int | float = 1.0,
                 bootstrap: bool = False,
                 n_jobs: int | None = -1,
                 random_state: int | RandomState | None = None):
        """Create an instance of :class:`IFAnomalyDetector`.

        :param n_trees: number of trees in the forest, defaults to 100.
        :type n_trees: `int`
        :param max_samples: number of samples for each tree, defaults to `auto`.
        :type max_samples: : `"auto"` or `int` or `float`
        :param contamination: expeted ratio of outliers, defaults to `auto`.
        :type contamination: `"auto"` or `float`
        :param max_features: number of features to use when splitting for each tree. If less than 1.0 then all features will be used, defaults to `1.0`.
        :type max_features: `int` or `float`
        :param bootstrap: if to reuse samples for trees. If False samples are unique per tree, defaults to `False`.
        :type bootstrap: `bool`
        :param n_jobs: number of parallel jobs/threads. If -1 uses all cores, defaults to `None`.
        :type n_jobs: `int`
        :param random_state: set a random state for results reproducibility, defaults to `None`.
        :type random_state: `int` or :class:`RandomState` or `None`
        """

        self.n_trees = n_trees
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def __repr__(self, **kwargs):
        return f'IFAnomalyDetector(trees={self.n_trees}, max_samples={self.max_samples}, contamination={self.contamination}, max_features={self.max_features}, bootstrap={self.bootstrap}, n_jobs={self.n_jobs})'

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return (
            isinstance(self.random_state, int) and
            isinstance(other.random_state, int) and
            self.get_params() == other.get_params()
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, warm_start: bool = False) -> Self:
        r"""Train the anomaly detector to be able to get the anomaly score for each data point

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the
          same length
        :param y: Membership for the vectors in `X`, defaults to `None`.
        :type y: iterable of `float` having the same length of `X`
        :param warm_start: flag triggering the non reinitialization of
          independent variables of the optimization problem, defaults to
          `None`.
        :type warm_start: `bool`
        :raises: RuntimeWarning if the solver found a suboptimal solution
        :raises: RuntimeError if the solver ended with an error
        :returns: self -- the trained model.
        """

        X, y = super()._check_X_y(X, y)

        rand_st = check_random_state(self.random_state)

        iso_forest = IsolationForest(n_estimators=self.n_trees, max_samples=self.max_features, contamination=self.contamination, max_features=self.max_features, bootstrap=self.bootstrap, n_jobs=self.n_jobs, random_state=rand_st, warm_start=warm_start)
        iso_forest.fit(np.array(X))

        self._forest = iso_forest
        self.score_05_ = .5

        return self

    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        The lower the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: array with the score for each vector of data
        """
        return self._forest.score_samples(X)

    def anomaly_score(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Get the anomaly scores for each of the original points data
        Is used by the fuzzifier to compute the membership for each data point.
        The bigger the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- of anomaly scores, between 0 and 1.
        """
        anomaly_scores = -self._forest.score_samples(X)
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        distances = (anomaly_scores - min_score) / (max_score - min_score)
        return distances

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Shifts the `score_samples` score to be able to then predict.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- for each value: less than zero is more anomalous, more than zero is more inlier.
        """
        return self._forest.decision_function(X)

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """
        Returns the array of predictions for each data point in X.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length.
        :returns: np.array -- for each value: -1 means anomalous, +1 means normal.
        """
        return self._forest.predict(X)


class LOFAnomalyDetector(AnomalyDetector):
    def __init__(self,
                 n_neighbors: int = 20,
                 algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute']='auto',
                 leaf_size: int = 30,
                 metric: str | Callable = 'minkowski',
                 p: int = 2,
                 metric_params: dict | None = None,
                 contamination: Literal['auto'] | float = 'auto',
                 novelty: bool = True,
                 n_jobs: int | None = -1):
        """Create an instance of :class:`LOFFuzzyInductor`.

        :param n_neighbors: the number of neighbors to consider when calculating local density, defaults to `20`.
        :type n_neighbors: `int`
        :param algorithm: algorithm used to compute the nearest neighbors, defaults to `auto`.
        :type algorithm: `Literal['auto', 'ball_tree', 'kd_tree', 'brute']`
        :param leaf_size: controls leaf size in tree-based algorithms BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem, defaults to 30.
        :type leaf_size: `int`
        :param metric: the distance metric to use, defaults to `'minkowski'` which results in the standard Euclidean distance when p = 2
        :type metric: `str` or `callable`
        :param p: parameter for the Minkowski metric from sklearn.metrics.pairwise_distances. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used, defautls to 2.
        :type p: `float`
        :param metric_params: additional keyword arguments for the metric function, defaults to `None`
        :type metric_params: `dict`
        :param contamination: expeted ratio of outliers. When fitting this is used to define the threshold on the scores of the samples, defaults to `auto`.
        :type contamination: `'auto'` or `float`
        :param novelty: by default, LocalOutlierFactor is only meant to be used for outlier detection (novelty=False). Set novelty to True if you want to use LocalOutlierFactor for novelty detection. In this case be aware that you should only use predict, decision_function and score_samples on new unseen data and not on the training set; and note that the results obtained this way may differ from the standard LOF results, defaults to `True`
        :type novelty: `bool`
        :param n_jobs: number of parallel jobs/threads. If -1 uses all cores, defaults to `None`.
        :type n_jobs: `int`
        """

        self.n_neighbors = n_neighbors
        self.algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs

    def __repr__(self, **kwargs):
        return f'LOFFuzzyInductor(n_neighbors={self.n_neighbors}, algorithm={self.algorithm}, leaf_size={self.leaf_size}, metric={self.metric}, p={self.p}, metric_params={self.metric_params}, contamination={self.contamination}, novelty={self.novelty}, n_jobs={self.n_jobs})'

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return self.get_params() == other.get_params()

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        r"""Train the anomaly detector to be able to get the anomaly score for each data point

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Membership for the vectors in `X`, defaults to `None`.
        :type y: iterable of `float` having the same length of `X`
        :returns: self -- the trained model.
        """

        X, y = super()._check_X_y(X, y)

        lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, algorithm=self.algorithm, leaf_size=self.leaf_size, metric=self.metric, p=self.p, metric_params=self.metric_params, contamination=self.contamination, novelty=self.novelty, n_jobs=self.n_jobs)

        lof.fit(np.array(X))

        self._lof = lof
        self.score_05_ = .5

        return self

    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Is the opposite of the Local Outlier Factor of X.
        The lower the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: array with the score for each vector of data
        """
        return self._lof.score_samples(X)

    def anomaly_score(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Get the anomaly scores for each of the original points data
        Is used by the fuzzifier to compute the membership for each data point.
        The bigger the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- of anomaly scores, between 0 and 1.
        """
        anomaly_scores = -self._lof.score_samples(X)
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        distances = (anomaly_scores - min_score) / (max_score - min_score)
        return distances

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Shifts the `score_samples` score to be able to then predict.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- for each value: less than zero is more anomalous, more than zero is more inlier.
        """
        return self._lof.decision_function(X)

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """
        Returns the array of predictions for each data point in X.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length.
        :returns: np.array -- for each value: -1 means anomalous, +1 means normal.
        """
        return self._lof.predict(X)
