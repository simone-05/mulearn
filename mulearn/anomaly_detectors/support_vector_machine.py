import logging

from typing import Self
from numpy.typing import NDArray, ArrayLike

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import mulearn.kernel as kernel
from mulearn.optimization import Solver, GurobiSolver


logger = logging.getLogger(__name__)


class SupportVectorMachine:
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
        isfitted_self = hasattr(self, 'chis_')
        isfitted_other = hasattr(other, 'chis_')
        if isfitted_self and isfitted_other:
            return equal and (self.chis_ == other.chis_)
        elif isfitted_self != isfitted_other:
            return False
        return equal

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, warm_start: bool = False) -> tuple[Self, float]:

        self._original_X = X.copy()

        if y is None:
            # Unsupervised version
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

        return self, self.score_05_

    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        X = np.array(X)
        t1 = self.k.compute(X, X)
        t2 = np.array([self.k.compute(x_i, X)
                        for x_i in self._original_X]).transpose().dot(self.chis_)
        ret = t1 -2 * t2 + self.fixed_term_
        return ret

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        return self.squared_radius_ - self.score_samples(X)

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        predictions = [1 if value>=0 else -1 for value in self.decision_function(X)]
        return np.array(predictions)