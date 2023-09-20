import numpy as np

# Import helper functions
from scratch_implemented.utils import bootstrap_samples
from scratch_implemented.statistical_machine_learning import ClassificationTree, RegressionTree
from collections import Counter


class RandomForest:
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.

    Parameters:
    -----------
    _n_trees: int
        The number of trees that are used.
    _max_features: int
        The maximum number of features that the trees are allowed to
        use.
    _min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    _min_impurity: float
        The minimum impurity required to split the tree further.
    _max_depth: int
        The maximum depth of a tree.

    TODO:

    apply multithreading
    """

    def __init__(
        self, _n_trees, _max_depth, _min_samles_split, _n_rand_features, _min_impurity
    ):
        self.n_trees = _n_trees
        self.max_depth = _max_depth
        self.min_samles_split = _min_samles_split
        self.n_rand_features = _n_rand_features
        self.min_impurity = _min_impurity
        self.trees = []


class RandomForestRegressor(RandomForest):
    def __init__(
        self,
        _n_trees=10,
        _max_depth=10,
        _min_impurity=1e-7,
        _min_samles_split=2,
        _n_rand_features=None,
    ):
        super(RandomForestRegressor, self).__init__(
            _n_trees=_n_trees,
            _max_depth=_max_depth,
            _min_samles_split=_min_samles_split,
            _n_rand_features=_n_rand_features,
            _min_impurity=_min_impurity,
        )

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = RegressionTree(
                _min_samples_split=self.min_samles_split,
                _max_depth=self.max_depth,
                _n_rand_features=self.n_rand_features,
                _min_impurity=self.min_impurity,
            )

            X_sample, y_sample = bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

    def predict(self, X):
        # predictions for each sample in each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])

        tree_predictions = np.swapaxes(predictions, 0, 1)

        return [np.mean(prediction) for prediction in tree_predictions]


class RandomForestClassifier:
    def __init__(
        self,
        _n_trees=10,
        _max_depth=10,
        _min_impurity=1e-7,
        _min_samles_split=2,
        _n_rand_features=None,
        _criterion="entropy",
    ):
        super(RandomForestClassifier, self).__init__(
            _n_trees=_n_trees,
            _max_depth=_max_depth,
            _min_samles_split=_min_samles_split,
            _n_rand_features=_n_rand_features,
            _min_impurity=_min_impurity,
        )
        self.criterion = _criterion

    def fit(self):
        for _ in range(self.n_trees):
            tree = ClassificationTree(
                _min_samples_split=self.min_samles_split,
                _max_depth=self.max_depth,
                _n_rand_features=self.n_rand_features,
                _min_impurity=self._min_impurity,
                _random_feature_selection=True,
                _criterion=self._criterion,
            )

            X_sample, y_sample = bootstrap_samples()
            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

    def _most_common_label(self, y):
        return Counter(y).most_common[1][0][0]

    def predict(self, X):
        # predictions for each sample in each tree
        predictions = np.array(
            [tree.predict(X) for tree in self.trees]
        )  # [[1,1,1,0],[0,0,0,1],[1,1,1,0]]

        tree_predictions = np.swapaxes(predictions, 0, 1)  # [[1,1,1,0]]

        # [1,1,1,0]
        # [0,0,0,1]
        # [1,1,1,0]]
        # (1 1 1 0) prediction for same sample from all three trees(magority vote)

        # We got (1 1 1 0) by means of magority vote
        return np.array(
            self._most_common_label(prediction) for prediction in tree_predictions
        )
