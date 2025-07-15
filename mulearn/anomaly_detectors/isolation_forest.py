import warnings
from typing import Any, Callable, Literal, Self
from numpy.typing import NDArray, ArrayLike

import math
import numpy as np
from functools import partial
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TreeNode:
    def __init__(self,
                 left: Self | None = None,
                 right: Self | None = None,
                 value: Any = None,
                 feature_index: int | None = None,
                 depth: int | None = None
                 ) -> None:
        self._left = left
        self._right = right
        self._value = value
        self._feature_index = feature_index
        self._depth = depth

    def __repr__(self) -> str:
        return f"TreeNode(feature_index: {self._feature_index}, value: {self._value}, depth: {self._depth})"

    def get_value(self) -> Any:
        return self._value

    def get_feat_idx(self) -> int | None:
        return self._feature_index

    def get_depth(self) -> int:
        return self._depth or 0

    def get_child(self, side: Literal["left", "right"]) -> Self | None:
        if side == "left":
        #     if self._left is None:
        #         raise Exception("Node has no left child")
            return self._left
        else:
            # if self._right is None:
        #         raise Exception("Node has no right child")
            return self._right

    def print_subTree(self, side: str = "", depth: int = 0) -> None:
        print(f"Depth {depth}:", "   " * depth, side, str(self.get_feat_idx()), str(self.get_value()))
        if self._left is not None:
            self._left.print_subTree("l", depth + 1)
        if self._right is not None:
            self._right.print_subTree("r", depth + 1)

class IsolationTree:
    def __init__(self,
                 max_depth: int = 20,
                 random_state: int | None = None
                 ) -> None:
        self.max_depth = max_depth
        self._random_state = random_state

    def _check_fitted(self) -> bool:
        if getattr(self, "_num_nodes", 0) == 1:
            warnings.warn("Tree has only the root node, try feeding more data", UserWarning)
        return (
            getattr(self, "_root", None) is not None and
            getattr(self, "_num_nodes", 0) > 0
        )

    def _build_splitting_function(self, X: NDArray, y: NDArray, verbose: int = 0) -> tuple[Callable[[NDArray[np.float64 | np.int_] | float | int], NDArray[np.float64]], bool]:
        """Is the inverse function of a cumulative distribution function.
        First return value is the inverse callable function, second is True if the cumulative is constant and thus returning the maximum value"""

        # X, y = check_X_y(X.reshape(1,-1), y)

        # Calculating the intervals in feature space:
        midpoints = (X[:-1] + X[1:]) / 2
        starts = np.concatenate(([X[0]], midpoints))
        ends = np.concatenate((midpoints, [X[-1]]))
        x_neighborhoods = np.stack((starts, ends), axis=1) # [[start,end], ...]
        y = 1 - y               # ones complement for y
        # Calculating areas:
        bases = x_neighborhoods[:, 1] - x_neighborhoods[:, 0]
        areas = bases * y       # base * height
        # Beta factor is so that: original_area = normalized_area / beta
        if areas.sum() == 0:
            if verbose >= 2:
                print(f"The cumulative function obtained is constant. Returning the value of the right extreme: {X[-1]}")
            def return_max_point(x):
                return np.full_like(x, X[-1])
            return return_max_point, True
        self.beta_ = 1 / areas.sum()
        normalized_areas = areas / areas.sum()    # normalize areas
        # Building the cumulative:
        # y0 = 0
        # y2 = 0 + normalized_area[0] + normalized_area[1]
        # yn = 1
        cumulative_areas = 0
        cumulative_xy = []      # [[x1,y1], .. [xn,yn]]
        for i,j in zip(x_neighborhoods, normalized_areas):
            cumulative_xy.append([i[0], cumulative_areas])
            cumulative_areas += j
            cumulative_xy.append([i[1], cumulative_areas])
        cumulative_xy[-1][1] = 1        # hardcode to be fully '1' (because it can be ~.999..)
        cumulative_xy = np.asarray(cumulative_xy)
        cumulative_xy = np.unique(cumulative_xy, axis=0)    # delete same pairs
        # Calculating the inverse of the cumulative function:
        # So that x belongs to [0.0, 1.0] and y belongs to original feature space (X)
        inv_big_f = partial(np.interp, xp=cumulative_xy[:,1], fp=cumulative_xy[:,0])

        if verbose >= 2:
            import plotly.graph_objects as go
            x_axis = np.linspace(0,1,1000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=inv_big_f(x_axis),
                mode="lines"
                )
            )
            fig.add_trace(go.Scatter(x=cumulative_xy[:,1],y=cumulative_xy[:,0],mode="markers",name="probability change"))
            fig.show()

        return inv_big_f, False

    def _build_subTree(self, start: int, end: int, depth: int = 0, verbose: int = 0) -> TreeNode | None:
        # Increase tree depth and nodes number
        if verbose == 2:
            print("Indexes interval start-end:",start,end)
        self._depth = max(self._depth, depth)
        self._num_nodes += 1

        # Sampling the feature index
        split_feature_index = self._random_number_generator.integers(self._X.shape[1])

        # Ordering based on the selected feature:
        # 'ord_feat_X' will be the current array where to operate, with start and end set, just one feature, and ordered
        # 'ord_y' will be the labels, with start and end set, with same ordering of 'ord_feat_X'
        sorting_indices = self._X.argsort(axis=0)[:,split_feature_index]
        ord_feat_X = self._X[sorting_indices, split_feature_index]
        ord_y = self._y[sorting_indices]

        if verbose == 2:
            print("Split feature index:", split_feature_index)

        # Returning if max_depth reached or just one node left
        if depth >= self.max_depth or end - start <= 1:
            if verbose == 2:
                if depth >= self.max_depth:
                    print("Max depth reached")
                if end>start:
                    print("Single node left, with data=", ord_feat_X[start])
                else:
                    print("Returning node because end<=start")
            return TreeNode(feature_index=split_feature_index, value=ord_feat_X[start] if end > start else None, depth=depth)

        min_value = ord_feat_X[start]
        max_value = ord_feat_X[end-1]
        if verbose == 2:
            print("Values interval:",min_value,max_value)

        if min_value == max_value:
            if verbose == 2:
                print("Returning node because minval == maxval, with data=",min_value)
            return TreeNode(feature_index=split_feature_index, value=min_value, depth=depth)

        # Build the splitting function on the interval
        splitting_func, constant_flag = self._build_splitting_function(ord_feat_X[start:end], ord_y[start:end], verbose=verbose)
        if constant_flag:
            if verbose == 2:
                print("Returning lowest value node because of constant cumulative function. Data=", min_value)
            return TreeNode(feature_index=split_feature_index, value=min_value, depth=depth)
        # Get split feature value and index
        split_value = splitting_func(self._random_number_generator.random())
        split_index = np.searchsorted(ord_feat_X[start:end], split_value, side='right') + start
        if verbose == 2:
            print("Split value:", split_value, "split index:", split_index)

        # Recursion on children:
        if verbose == 2:
            print("continuing left" if split_index>start else "no nodes left to the left")
            print("continuing right" if split_index<end else "no nodes left to the right")
        left_child = self._build_subTree(start, split_index, depth + 1, verbose=verbose) if split_index > start else None
        right_child = self._build_subTree(split_index, end, depth + 1, verbose=verbose) if split_index < end else None

        return TreeNode(feature_index=split_feature_index, value=split_value, left=left_child, right=right_child, depth=depth)

    def fit(self, X: ArrayLike, y: ArrayLike, verbose: int = 0) -> Self:
        self._X = X.copy()
        self._y = y.copy()
        self._depth = 0
        self._num_nodes = 0
        self._random_number_generator = np.random.default_rng(self._random_state)
        self._root = self._build_subTree(0, len(self._X), verbose=verbose)
        return self

    def get_node_depth(self, target_data) -> int:
        target_data = np.asarray(target_data)
        if target_data.ndim > 1:
            raise TypeError(f"target_data must have dimension=1, but it has dimension={target_data.ndim}")
        if not self._check_fitted():
            raise RuntimeError("Tree not fitted")
        cur_node = self._root
        while True:
            cur_feat_idx = cur_node.get_feat_idx()
            cur_data = cur_node.get_value()
            cur_depth = cur_node.get_depth()
            if target_data[cur_feat_idx] > cur_data:
                    cur_node = cur_node.get_child("right")
            else:
                    cur_node = cur_node.get_child("left")
            if cur_node is None:
                break
        return cur_depth

    def get_max_depth(self) -> int:
        return self._depth

    def get_num_nodes(self) -> int:
        return self._num_nodes

    def print_tree(self) -> None:
        if self._root is None:
            print("Empty tree")
            return
        self._root.print_subTree()

class SupervisedIsolationForest:
    def __init__(self,
                 n_estimators: int = 100,
                 max_samples: Literal["auto"] | int | float = 256,
                 max_features: int | float = 1.0,
                 max_depth: int = 100,
                 random_state: int | None = None
                 ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._max_samples = max_samples
        self._max_features = max_features
        self._random_state = random_state

    def _check_fitted(self):
        return (
            getattr(self, "forest_", None) is not None and
            len(self.forest_) > 0
        )

    def fit(self, X: ArrayLike, y: ArrayLike, verbose: int = 0) -> Self:

        X, y = check_X_y(X, y)
        X, y = np.asarray(X), np.asarray(y)
        self._X, self._y = X.copy(), y.copy()

        dataset_size = len(self._X)
        dataset_features_number = self._X.shape[1]

        # 'max_samples' parameter check and setting:
        if self._max_samples == "auto":
            max_samples = 256
        elif type(self._max_samples) == float:
            maxs = self._max_samples
            if maxs > 1. or maxs < .0:
                raise ValueError(f"if max_samples is a float, it should be between 0.0 and 1.0. Given value: {maxs}")
            max_samples = dataset_size * maxs
        elif type(self._max_samples) == int:
            max_samples = self._max_samples
        else:
            raise ValueError(f"max_samples should be either a 'int', 'float' or string value \"auto\", given: {type(self._max_samples)}")
        max_samples = min(max_samples, dataset_size)

        # 'max_features' parameter check and setting:
        if type(self._max_features) == int:
            max_features = self._max_features
        elif type(self._max_features) == float:
            maxf = self._max_features
            if maxf > 1. or maxf < .0:
                raise ValueError(f"if max_features is a float, it should be between 0.0 and 1.0. Given value: {maxf}")
            max_features = int(dataset_features_number * maxf) or 1
        else:
            raise ValueError(f"max_features should be either a 'int' or 'float', given: {type(self._max_features)}")
        max_features = min(max_features, dataset_features_number)

        self._random_number_generator = np.random.default_rng(self._random_state)

        if verbose == 2:
            import time
            start = time.time()
        if verbose >= 1:
            from tqdm import tqdm
            v = [_ for _ in range(self._n_estimators)]
            trees_iter = tqdm(v, desc=f'Fitting {self._n_estimators} trees')
        else:
            trees_iter = range(self._n_estimators)

        self.forest_ = []

        # Fitting trees:
        for _ in trees_iter:

            # # Selecting random feature from X:
            # selected_feature_index = self._random_number_generator.integers(0, num_features)
            # self._trees_feature_indexes.append(selected_feature_index)
            # indexer = [slice(None)] * X.ndim
            # indexer[1] = selected_feature_index
            # single_feature_X = X[tuple(indexer)]

            # # Subsampling from (X,y):
            samples_indeces = np.sort(self._random_number_generator.permutation(dataset_size)[:max_samples])
            sampled_X = X[samples_indeces]
            sampled_y = y[samples_indeces]

            # # Feature setting for the samples:
            sampled_features = np.sort(self._random_number_generator.choice(dataset_features_number, size=max_features, replace=False))
            sampled_X[:,sampled_features]

            # Training a tree:
            tree = IsolationTree(max_depth=self._max_depth)
            tree.fit(sampled_X, sampled_y, verbose=verbose-1)
            self.forest_.append(tree)

        # Saving 'c(n)' term to later calculate the score for any data point:
        # 'dataset_size' is 'n'
        self.c_of_n_ = 2 * (math.log(dataset_size-1) + np.euler_gamma) - (2*(dataset_size-1)/dataset_size)

        if verbose == 2:
            print(time.time()-start,"seconds")

        return self

    def score_samples(self, X: ArrayLike, verbose: bool = False) -> NDArray[np.float64]:
        """ Higher -> inlier. Lower -> outlier """
        if verbose:
            import time
            start = time.time()
        res = []
        for sample in X:
            avg_depth = 0
            for i, tree in enumerate(self.forest_):
                # Selecting tree's feature:
                # selected_feature_index = self._trees_feature_indexes[i]
                # indexer = [slice(None)]
                # indexer[0] = selected_feature_index
                # single_feature_sample = sample[tuple(indexer)]
                avg_depth += tree.get_node_depth(sample)
            avg_depth = avg_depth/self._n_estimators
            # Original sklearn's IsolationForest formula:
            score = 2 ** (- avg_depth / self.c_of_n_)
            # For conformity of other anomaly detectors i complement the score:
            res.append(1- score)
        if verbose:
            print(time.time()-start,"seconds")
        return np.asarray(res)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """ Shifts the score_samples to be between -1 and 1. The lower the more abnormal """
        return self.score_samples(X) - .5

    def anomaly_score(self, X: ArrayLike) -> NDArray[np.float64]:
        anomaly_scores = -self.score_samples(X)
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        distances = (anomaly_scores - min_score) / (max_score - min_score)
        return distances

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """ Array of -1 if anomalous, +1 if non-anomaly """
        return np.where(self.score_samples(X) > 0, 1, -1)