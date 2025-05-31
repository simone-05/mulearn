__version__ = '1.1.3'


import copy
import logging
from typing import Iterable, Self, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.optimize import OptimizeWarning
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import mulearn.fuzzifier as fuzzifier
import mulearn.anomaly_detector as anomaly_detector

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=OptimizeWarning)
warnings.filterwarnings('ignore', category=FitFailedWarning)


class FuzzyInductor(BaseEstimator, RegressorMixin):
    """Base class for FuzzyInductor."""

    def __init__(self,
                 fuzzifier: fuzzifier.Fuzzifier = fuzzifier.ExponentialFuzzifier(),
                 anomaly_detector: anomaly_detector.AnomalyDetector = anomaly_detector.SVMAnomalyDetector(),
                 keep_original_data: bool = False) -> None :
        r"""Create an instance of :class:`FuzzyInductor`.

        :param fuzzifier: fuzzifier mapping distance values to membership degrees, defaults to `ExponentialFuzzifier()`.
        :type fuzzifier: :class:`mulearn.fuzzifier.Fuzzifier`
        :param anomaly_detector: the anomaly detector implementation to use, defaults to `SVMAnomalyDetector()`.
        :type anomaly_detector: :class:mulearn.anomaly_detector.AnomalyDetector`
        :param save_data: specifies if we want the model to save the original data to visualize the profile of the membership later, defaults to `False`.
        :type save_data: `bool`
        """

        self.fuzzifier = fuzzifier
        self.anomaly_detector = anomaly_detector
        self.keep_original_data = keep_original_data

    def __repr__(self, **kwargs):
        raise NotImplementedError(
            'The base class does not implement the `__repr__` method')

    def __eq__(self, other):
        raise NotImplementedError(
            'The base class does not implement the `__eq__` method')

    def fit(self, X: ArrayLike , y: ArrayLike | None = None, **kwargs) -> Self:
        r""" Train the anomaly detector to be able to get the anomaly score for each data point.
        
        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length.
        :param y: Labels containing the *gold standard* membership values for the vectors in `X`.
        :type y: iterable of `float` having the same length of `X`
        :raises: ValueError if the values in `y` are not between 0 and 1, if `X` and have different lengths, or if `X` contains elements of different lengths.
        :returns: self -- the trained model.
        """

        if self.keep_original_data:
            self.X_ = X

        self.anomaly_detector.fit(X, y, **kwargs)

        self.fuzzifier.fit(
            self.anomaly_detector.anomaly_score(X),
            y,
            self.anomaly_detector.score_05
        )

        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float_]:
        r"""Compute predictions for the membership function.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the
          same length
        :returns: array of float -- the predictions for each value in `X`.
        """
        X = check_array(X)
        return self.fuzzifier.get_membership(self.anomaly_detector.anomaly_score(X))

    def predict(self, X: ArrayLike, alpha: float | None = None) -> Iterable[int|float]:
        r"""Compute predictions for membership to the set.

        Predictions are either computed through the membership function (when
        `alpha` is set to a float in [0, 1]) or obtained via an $\alpha$-cut on
        the same function (when `alpha` is set to `None`).

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param alpha: $\alpha$-cut value, defaults to `None`.
        :type alpha: float
        :raises: ValueError if `alpha` is set to a value different from
          `None` and not included in $[0, 1]$.
        :returns: array -- the predictions for each value in `X`.
        """
        X = check_array(X)
        mus = self.decision_function(X)
        if alpha is None:
            return mus
        else:
            if alpha < 0 or alpha > 1:
                raise ValueError('alpha cut value should belong to [0, 1]'
                                 f' (provided {alpha})')
            return np.array([1 if mu >= alpha else 0 for mu in mus])

    def score(self, X: ArrayLike, y: ArrayLike, **kwargs) -> np.float_:
        r"""Compute the fuzzifier score.

        Score is obtained as the opposite of MSE between predicted
        membership values and labels.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Labels containing the *gold standard* membership values
          for the vectors in `X`.
        :type y: iterable of `float` having the same length of `X`
        :returns: `float` -- opposite of MSE between the predictions for the
          elements in `X` w.r.t. the labels in `y`.
        """
        X, y = check_X_y(X, y)

        return -np.mean((self.decision_function(X) - y) ** 2)

    def get_profile(self) -> list:
        if not self.keep_original_data:
            raise RuntimeError('Cannot execute the method `get_profile` without specifying `keep_original_data=True` in the constructor\'s parameters')
        return self.fuzzifier.get_profile(self.anomaly_detector.anomaly_score(self.X_))
