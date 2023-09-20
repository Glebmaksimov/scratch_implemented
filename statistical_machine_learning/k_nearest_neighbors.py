from collections import Counter
import numpy as np
import math
from utils.metrics import (
    manghattan_distance,
    minkowski_distance,
    hamming_distance,
    cosine_distance,
    euclidian_distance,
)


class KNN(object):
    def __init__(self, _k, _distance_metric, _weights, _weighter):
        self.k = _k
        self.weights = _weights
        distance_metrics = {
            "manghattan": manghattan_distance,
            "minkowski": minkowski_distance,
            "hamming": hamming_distance,
            "cosine": cosine_distance,
        }
        self.weighter = _weighter
        self.distance_metric = distance_metrics.get(
            _distance_metric, euclidian_distance
        )

    def fit(self, X, y):
        self.X = X
        self.y = y

    def _weigh(self, distance, n=0.25):
        weighters = {
            "exponent": math.pow(n, distance),
        }
        return weighters.get(self.weighter, 1 / (distance + 1))

    def _get_labels_weights(self, distances):
        weights = [self._weigh(d) for d in distances]
        return weights

    def _get_knn_labels(self, x):
        distances = [self.distance_metric(x, x_tr) for x_tr in self.X]
        knn_indices = np.argsort(distances)[: self.k]
        knn_labels = [self.y[i] for i in knn_indices]
        return distances, knn_labels


class KNNClassifier(KNN):
    """K Nearest Neighbors Regressor.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the
        sample that we wish to predict.

    """

    def __init__(
        self,
        _k=5,
        _distance_metric="euclidian",
        _weights="uniform",
        _weighter=None,
    ):
        super(KNNClassifier, self).__init__(
            _k=_k,
            _distance_metric=_distance_metric,
            _weights=_weights,
            _weighter=_weighter,
        )

    def fit(self, X, y):
        super(KNNClassifier, self).fit(X, y)

    def _predict(self, x):
        distances, knn_labels = super(KNNClassifier, self)._get_knn_labels(x)

        # if we need not weighting | to add weights set to "distance"
        if self.weights == "uniform":
            most_common = Counter(knn_labels).most_common()
            return most_common[0][0]

        unique_labels = np.unique(knn_labels)
        weight_per_label = {key: 0 for key in unique_labels}
        weights = self._get_labels_weights(distances)
        for label, weight in (unique_labels, weights):
            weight_per_label[label] += weight

        return max(weight_per_label, key=weight_per_label.get)

    def predict(self, X):
        return [self._predict(x) for x in X]


class KNNRegressor(KNN):
    """K Nearest Neighbors Regressor.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the
        sample that we wish to predict.

    -----------
    References:

    https://www.analyticsvidhya.com/blog/2021/08/how-knn-uses-distance-measures/ - dostance metrics

    https://www.youtube.com/watch?v=m_bWhOLr_XM&t=0s - KNN

    https://www.youtube.com/watch?v=s5Ms80gpbmA&t=542s - weighted KNN

    """

    def __init__(
        self,
        _k=5,
        _distance_metric="euclidian",
        _weights="uniform",
        _weighter=None,
    ):
        super(KNNRegressor, self).__init__(
            _k=_k,
            _distance_metric=_distance_metric,
            _weights=_weights,
            _weighter=_weighter,
        )

    def fit(self, X, y):
        super(KNNRegressor, self).fit(X, y)

    def _predict(self, x):
        distances, knn_labels = super(KNNRegressor, self)._get_knn_labels(x)

        # if we need not weighting | to add weights set to "distance"
        if self.weights == "uniform":
            return np.mean(knn_labels)

        weights = self._get_labels_weights(distances)
        predicted_value = np.sum(
            [(label * weight) for label, weight in zip(knn_labels, weights)]
        ) / np.sum(weights)
        return predicted_value

    def predict(self, X):
        return [self._predict(x) for x in X]
