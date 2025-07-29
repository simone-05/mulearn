import logging

from typing import Self
from numpy.typing import NDArray, ArrayLike

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

import mulearn.kernel as kernel
from mulearn.optimization import Solver, GurobiSolver


logger = logging.getLogger(__name__)


class SupportVectorMachine(RegressorMixin, BaseEstimator):
    def __init__(self,
                 c: float = 1,
                 k: kernel.Kernel = kernel.GaussianKernel(),
                 solver: Solver = GurobiSolver()
                 ) -> None:
        self.c = c
        self.k = k
        self.solver = solver

    def __eq__(self, other):
        """Check equality w.r.t. other objects."""
        equal = (
            type(self) == type(other) and
            self.get_params() == other.get_params()
        )
        isfitted_self = hasattr(self, "chis_")
        isfitted_other = hasattr(other, "chis_")
        if isfitted_self and isfitted_other:
            return equal and (self.chis_ == other.chis_)
        elif isfitted_self != isfitted_other:
            return False
        return equal

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, warm_start: bool = False) -> Self:
        """Induce the membership function starting from a labeled sample.

        :param X: vectors in data space
        :type X: ArrayLike
        :param y: membership for the vectors in `X`, defaults to None for the unsupervised version
        :type y: ArrayLike | None, optional
        :param warm_start: flag triggering the non reinitialization of
          independent variables of the optimization problem, defaults to False
        :type warm_start: bool, optional
        :raises ValueError: if the values in `y` are not between 0 and 1, if
          `X` and have different lengths, or if `X` contains elements of
          different lengths.
        :raises NotFittedError: if we set warm_start = True but the model hasn't been fit
        :return: the trained model
        :rtype: Self
        """
        X = check_array(X)

        if y is None:
            # Unsupervised version
            y = np.ones(len(X))
        else:
            for e in y:
                if e < 0 or e > 1:
                    raise ValueError("`y` values should belong to [0, 1]")

        self._original_X = X.copy()
        X, y = check_X_y(X, y)

        if warm_start:
            check_is_fitted(self, ["chis_"])
            if self.chis_ is None:
                raise NotFittedError("chis variable are set to None")
            self.solver.initial_values = self.chis_

        if type(self.k) is kernel.PrecomputedKernel:
            idx = X.flatten()
            self._gram = self.k.kernel_computations[idx][:, idx]
        else:
            self._gram = np.array([self.k.compute(x1, X) for x1 in X])

        self.chis_ = self.solver.solve(X, y, self.c, self._gram)

        self.fixed_term_ = np.array(self.chis_).dot(self._gram.dot(self.chis_))

        chi_SV_index = [i for i, (chi, mu) in enumerate(zip(self.chis_, y))
                        if -self.c * (1 - mu) < chi < self.c * mu]
        chi_sq_radius = list(self.anomaly_score(X[chi_SV_index]))

        if len(chi_sq_radius) == 0:
            self.chis_ = None
            logger.warning("No support vectors found")
            return self

        self.squared_radius_ = np.mean(chi_sq_radius)

        return self

    def anomaly_score(self, X: ArrayLike) -> NDArray[np.float64]:
        """Compute the distance associated to the samples. Higher is more anomalous

        :param X: data points
        :type X: ArrayLike
        :return: array with the squared distances of the data points
        :rtype: NDArray[np.float64]
        """
        check_is_fitted(self, ["chis_"])
        X = check_array(X)
        t1 = self.k.compute(X, X)
        t2 = np.array([self.k.compute(x_i, X)
                        for x_i in self._original_X]).transpose().dot(self.chis_)
        ret = t1 -2 * t2 + self.fixed_term_
        return ret

    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ["chis_"])
        X = check_array(X)
        return -self.anomaly_score(X)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Decision function derived from the score_samples. Lower is more anomalous

        :param X: data points
        :type X: ArrayLike
        :return: array of values: more negative for more anomalous, more positive for more normal points
        :rtype: NDArray[np.float64]
        """
        check_is_fitted(self, ["chis_"])
        X = check_array(X)
        return self.squared_radius_ - self.anomaly_score(X)

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """Prediction function. -1 for anomalies, +1 for normal points

        :param X: data points
        :type X: ArrayLike
        :return: array of predictions
        :rtype: NDArray[np.int_]
        """
        check_is_fitted(self, ["chis_"])
        X = check_array(X)

        predictions = [1 if value>=0 else 0 for value in self.decision_function(X)]
        return np.asarray(predictions)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Compute the score obtained as the inverted MSE between predicted
        membership values and labels.

        1.0 for best score, and 0.0 for worst score.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Labels containing the *gold standard* membership values
          for the vectors in `X`.
        :type y: iterable of `float` having the same length of `X`
        :returns: `float` -- inverted MSE between the predictions for the
          elements in `X` w.r.t. the labels in `y`.
        """
        X, y = check_X_y(X, y)
        return 1.0 - np.mean((self.predict(X) - y) ** 2)