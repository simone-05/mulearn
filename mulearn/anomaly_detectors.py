
"""This module implements the anomaly detectors used in mulearn.
"""

import logging
from typing import Iterable, Literal, Self, Sequence

import numpy as np
from numpy.typing import NDArray, ArrayLike
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state

import mulearn.kernel as kernel
from mulearn.optimization import GurobiSolver

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Base class for anomaly detectors.

    The base class for anomaly detectors is AnomalyDetector: it defines the signatures of the basic methods. 
    - `fit`, trains the anomaly detector to be able to get the anomaly score for each data point.
    - `get_membership`, returning the membership function inferred from data
    - `get_profile`, computing information exploitable in order to visualize the fuzzifier in graphical form.
    """

    def fit(self, X: ArrayLike, y: Sequence[int|float] | NDArray | None = None, **kwargs) -> Self:
        """Train the anomaly detector to be able to get the anomaly score for each data point

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Membership for the vectors in `X`, defaults to `None`.
        :type y: iterable of `float` having the same length of `X`
        :returns: self -- the trained model.
        """
        raise NotImplementedError(
            'The base class does not implement the `fit` method')

    def score_samples(self, X: ArrayLike) -> NDArray:
        """
        Is the most direct score that an implementation can give.
        The lower the more anomalous.
        
        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: array with the score for each vector of data
        """
        raise NotImplementedError(
            'The base class does not implement the `score_samples` method')

    def anomaly_score(self, X: ArrayLike) -> NDArray:
        """
        Get the anomaly scores for each of the original points data.
        The bigger the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- of anomaly scores, between 0 and 1.
        """
        raise NotImplementedError(
            'The base class does not implement the `anomaly_score` method')

    def decision_function(self, X: ArrayLike) -> NDArray:
        """
        Shifts the `score_samples` score to be able to then predict.
        
        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- for each value: less than zero is more anomalous, more than zero is more inlier.
        """
        raise NotImplementedError(
            'The base class does not implement the `decision_function` method')

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Returns the array of predictions for each data point in X.
        
        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length.
        :returns: np.array -- for each value: -1 means anomalous, +1 means normal.
        """
        raise NotImplementedError(
            'The base class does not implement the `predict` method')


class SVMAnomalyDetector(AnomalyDetector, BaseEstimator):
    def __init__(self,
                 c=1,
                 k=kernel.GaussianKernel(),
                 solver=GurobiSolver(),
                 random_state=None):
        """Create an instance of :class:`SVMAnomalyDetector`.

        :param c: Trade-off constant, defaults to 1.
        :type c: `float`
        :param k: Kernel function, defaults to :class:`GaussianKernel()`.
        :type k: :class:`mulearn.kernel.Kernel`
        :param solver: Solver to be used to obtain the optimization problem
          solution, defaults to `GurobiSolver()`.
        :type solver: :class:`mulearn.optimization.Solver`
        :param random_state: Seed of the pseudorandom generator.
        :type random_state: `int`
        """

        self.c = c
        self.k = k
        self.solver = solver
        self.random_state = random_state
        self.chis_ = None
        self.gram_ = None
        self.fixed_term_ = None
        self.X_ = None

    def __repr__(self, **kwargs):
        return f'SVMAnomalyDetector(c={self.c}, k={self.k}, ' \
               f'solver={self.solver})'

    def __eq__(self, other):
        """Check equality w.r.t. other objects."""
        equal = (type(self) is type(other) and \
                 self.c == other.c and self.k == other.k)
        if 'chis_' in self.__dict__:
            if 'chis_' not in other.__dict__:
                return False
            else:
                return equal and (self.chis_ == other.chis_)

    def fit(self, X: ArrayLike, y: Sequence[int|float] | NDArray | None = None, warm_start: bool = False) -> Self:
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
        :returns: self -- the trained model.
        """

        self.X_ = X

        if y is None:
            y = np.ones(len(X))

        self.random_state = check_random_state(self.random_state)

        if warm_start:
            check_is_fitted(self, ['chis_'])
            if self.chis_ is None:
                raise NotFittedError('chis variable are set to None')
            self.solver.initial_values = self.chis_

        if type(self.k) is kernel.PrecomputedKernel:
            idx = X.flatten()
            self.gram_ = self.k.kernel_computations[idx][:, idx]
        else:
            self.gram_ = np.array([self.k.compute(x1, X) for x1 in X])

        self.chis_ = self.solver.solve(X, y, self.c, self.gram_)

        self.fixed_term_ = np.array(self.chis_).dot(self.gram_.dot(self.chis_))

        chi_SV_index = [i for i, (chi, mu) in enumerate(zip(self.chis_, y))
                        if -self.c * (1 - mu) < chi < self.c * mu]
        chi_sq_radius = list(self.anomaly_score(X[chi_SV_index]))

        if len(chi_sq_radius) == 0:
            self.chis_ = None
            logger.warning('No support vectors found')
            return self

        self.squared_radius = np.mean(chi_sq_radius)

        return self

    def score_samples(self, X: ArrayLike) -> NDArray:
        #TODO - to implement
        """
        The lower the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: array with the score for each vector of data
        """
        raise NotImplementedError('Method not yet implemented')

    def anomaly_score(self, X: ArrayLike) -> NDArray:
        """
        Get the anomaly scores for each of the original points data
        The bigger the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- of anomaly scores, between 0 and 1.
        """
        X = np.array(X)
        t1 = self.k.compute(X, X)
        t2 = np.array([self.k.compute(x_i, X)
                        for x_i in self.X_]).transpose().dot(self.chis_)
        ret = t1 -2 * t2 + self.fixed_term_
        return ret

    def decision_function(self, X: ArrayLike) -> NDArray:
        #TODO - to implement
        """
        Shifts the `score_samples` score to be able to then predict.
        
        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- for each value: less than zero is more anomalous, more than zero is more inlier.
        """
        raise NotImplementedError('Method not yet implemented')

    def predict(self, X: ArrayLike) -> NDArray:
        #TODO - to implement
        """
        Returns the array of predictions for each data point in X.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length.
        :returns: np.array -- for each value: -1 means anomalous, +1 means normal.
        """
        raise NotImplementedError('Method not yet implemented')


class IFAnomalyDetector(AnomalyDetector):
    def __init__(self,
                 n_trees=100,
                 max_samples='auto',
                 contamination='auto',
                 max_features=1.0,
                 bootstrap=False,
                 n_jobs=-1,
                 random_state=None,
                 squared_radius=.5):
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
        self.squared_radius = squared_radius

    def __repr__(self, **kwargs):
        return f'IFAnomalyDetector(trees={self.n_trees}, max_samples={self.max_samples}, contamination={self.contamination}, max_features={self.max_features}, bootstrap={self.bootstrap}, n_jobs={self.n_jobs})'

    def fit(self, X: ArrayLike, y: Sequence[int|float] | NDArray | None = None, warm_start: bool = False) -> Self:
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
        :returns: self -- the trained model.
        """

        self.random_state = check_random_state(self.random_state)

        iso_forest = IsolationForest(n_estimators=self.n_trees, max_samples=self.max_features, contamination=self.contamination, max_features=self.max_features, bootstrap=self.bootstrap, n_jobs=self.n_jobs, random_state=self.random_state, warm_start=warm_start)
        iso_forest.fit(np.array(X))

        self.forest_ = iso_forest

        return self

    def score_samples(self, X: ArrayLike) -> NDArray:
        """
        The lower the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: array with the score for each vector of data
        """
        return self.forest_.score_samples(X)
    
    def anomaly_score(self, X: ArrayLike) -> NDArray:
        """
        Get the anomaly scores for each of the original points data
        The bigger the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- of anomaly scores, between 0 and 1.
        """
        anomaly_scores = -self.forest_.score_samples(X)
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        distances = (anomaly_scores - min_score) / (max_score - min_score)
        return distances

    def decision_function(self, X: ArrayLike) -> NDArray:
        """
        Shifts the `score_samples` score to be able to then predict.
        
        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- for each value: less than zero is more anomalous, more than zero is more inlier.
        """
        return self.forest_.decision_function(X)

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Returns the array of predictions for each data point in X.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length.
        :returns: np.array -- for each value: -1 means anomalous, +1 means normal.
        """
        return self.forest_.predict(X)


class LOFAnomalyDetector(AnomalyDetector):
    def __init__(self,
                 n_neighbors=20,
                 algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute']='auto',
                 leaf_size=30,
                 metric='minkowski',
                 p=2,
                 metric_params=None,
                 contamination='auto',
                 novelty=True,
                 n_jobs=-1,
                 squared_radius=.5):
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
        self.squared_radius = squared_radius

    def __repr__(self, **kwargs):
        return f'LOFFuzzyInductor(n_neighbors={self.n_neighbors}, algorithm={self.algorithm}, leaf_size={self.leaf_size}, metric={self.metric}, p={self.p}, metric_params={self.metric_params}, contamination={self.contamination}, novelty={self.novelty}, n_jobs={self.n_jobs})'

    def fit(self, X: ArrayLike, y: Sequence[int|float] | NDArray | None = None) -> Self:
        r"""Train the anomaly detector to be able to get the anomaly score for each data point

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Membership for the vectors in `X`, defaults to `None`.
        :type y: iterable of `float` having the same length of `X`
        :returns: self -- the trained model.
        """

        lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, algorithm=self.algorithm, leaf_size=self.leaf_size, metric=self.metric, p=self.p, metric_params=self.metric_params, contamination=self.contamination, novelty=self.novelty, n_jobs=self.n_jobs)

        lof.fit(np.array(X))

        self.lof_ = lof

        return self

    def score_samples(self, X: ArrayLike) -> NDArray:
        """
        Is the opposite of the Local Outlier Factor of X.
        The lower the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: array with the score for each vector of data
        """
        return self.lof_.score_samples(X)

    def anomaly_score(self, X: ArrayLike) -> NDArray:
        """
        Get the anomaly scores for each of the original points data
        The bigger the more anomalous.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- of anomaly scores, between 0 and 1.
        """
        anomaly_scores = -self.lof_.score_samples(X)
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        distances = (anomaly_scores - min_score) / (max_score - min_score)
        return distances

    def decision_function(self, X: ArrayLike) -> NDArray:
        """
        Shifts the `score_samples` score to be able to then predict.
        
        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: np.array -- for each value: less than zero is more anomalous, more than zero is more inlier.
        """
        return self.lof_.decision_function(X)

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Returns the array of predictions for each data point in X.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length.
        :returns: np.array -- for each value: -1 means anomalous, +1 means normal.
        """
        return self.lof_.predict(X)
