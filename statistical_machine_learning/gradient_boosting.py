import numpy as np

# Import helper functions
from scratch_implemented.utils import to_categorical
from scratch_implemented.deep_learning.loss_functions import SquareLoss, CrossEntropy
from scratch_implemented.statistical_machine_learning.decision_tree import RegressionTree


class GradientBoosting(object):
    """Super class of GradientBoostingClassifier and GradientBoostinRegressor.
    Uses a collection of regression trees that trains on predicting the gradient
    of the loss function.

    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    regression: boolean
        True or false depending on if we're doing regression or classification.
    """

    def __init__(
        self,
        _n_estimators,
        _learning_rate,
        _min_samples_split,
        _min_impurity,
        _max_depth,
        _regression,
        _n_rand_features,
    ):
        self.n_estimators = _n_estimators
        self.learning_rate = _learning_rate
        self.min_samples_split = _min_samples_split
        self.min_impurity = _min_impurity
        self.max_depth = _max_depth
        self.regression = _regression
        self.n_rand_features = _n_rand_features

        # Square loss for regression
        # Log loss for classification
        self.loss = SquareLoss()
        if not self.regression:
            self.loss = CrossEntropy()

        # Initialize regression trees
        self.trees = []
        for _ in range(_n_estimators):
            tree = RegressionTree(
                _min_samples_split=self.min_samples_split,
                _max_depth=self.max_depth,
                _n_rand_features=self.n_rand_features,
                _min_impurity=self.min_impurity,
            )
            self.trees.append(tree)

    def fit(self, X, y):
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for i in range(self.n_estimators):
            gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            # Update y prediction
            y_pred -= np.multiply(self.learning_rate, update)

    def predict(self, X):
        y_pred = np.array([])
        # Make predictions
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update

        if not self.regression:
            # Turn into probability distribution
            y_pred = np.exp(y_pred) / np.expand_dims(
                np.sum(np.exp(y_pred), axis=1), axis=1
            )
            # Set label to the value that maximizes probability
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GradientBoostingRegressor(GradientBoosting):
    def __init__(
        self,
        _n_estimators=4,
        _learning_rate=0.5,
        _min_samples_split=2,
        _min_var_red=1e-7,
        _max_depth=4,
        _n_rand_features=6,
    ):
        super(GradientBoostingRegressor, self).__init__(
            _n_estimators=_n_estimators,
            _learning_rate=_learning_rate,
            _min_samples_split=_min_samples_split,
            _min_impurity=_min_var_red,
            _max_depth=_max_depth,
            _regression=True,
            _n_rand_features=_n_rand_features,
        )


# https://ericwebsmith.github.io/2020/04/19/GradientBoostingClassification/


class GradientBoostingClassifier(GradientBoosting):
    def __init__(
        self,
        _n_estimators=4,
        _learning_rate=0.5,
        _min_samples_split=2,
        _min_info_gain=1e-7,
        _max_depth=4,
        _n_rand_features=6,
    ):
        super(GradientBoostingClassifier, self).__init__(
            _n_estimators=_n_estimators,
            _learning_rate=_learning_rate,
            _min_samples_split=_min_samples_split,
            _min_impurity=_min_info_gain,
            _max_depth=_max_depth,
            _n_rand_features=_n_rand_features,
            regression=False,
        )

    def fit(self, X, y):
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)
