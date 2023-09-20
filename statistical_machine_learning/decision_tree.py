import numpy as np

from scratch_implemented.utils import split_data_by_feature
from scratch_implemented.utils import (
    calculate_entropy,
    calculate_error_rate,
    calculate_gini_index,
    calculate_laplace_error_rate,
)


class Node:
    """Class that represents a decision node or leaf in the decision tree

    Parameters:
    -----------
    _feature_index: int
        Feature index which we want to use as the threshold measure.
    _threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    _value: float
        The class prediction if classification tree, or float value if regression tree.
    _left: DecisionNode
        Next decision node for samples where features value met the threshold.
    _right: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """

    def __init__(
        self,
        _feature_index=None,
        _threshold=None,
        _left=None,
        _right=None,
        *,
        _value=None
    ):
        self.feature_index = _feature_index  # Index for the feature that is tested
        self.threshold = _threshold  # Threshold value for feature
        self.left = _left  # 'Left' subtree
        self.right = _right  # 'Right' subtree
        self.value = _value  # Value if the node is a leaf in the tree

    def _is_leaf_node(self):
        return self.value is not None


# Super class of RegressionTree and ClassificationTree
class DecisionTree(object):
    """Super class of RegressionTree and ClassificationTree.

    Parameters:
    -----------
    _min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    _min_impurity: float
        The minimum impurity required to split the tree further.
    _max_depth: int
        The maximum depth of a tree.
    _loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.

    -----------
    References

    Post Pruning Example - https://www.jinhang.work/tech/decision-tree-in-python/

    Post Pruning Theory - https://www.youtube.com/watch?v=qTs-lLNFxc0&list=PLUZjIBGiCHFfRJwflq6NqU3CuiPhAhSfi&index=18

    MFTI - https://www.youtube.com/watch?v=vzbyk_7HdiQ&list=PL4_hYwCyhAvasRqzz4w562ce0esEwS0Mt&index=5

    Reduced Error Puning https://www.youtube.com/watch?v=u4kbPtiVVB8&t=0s

    Cost Complexity Pruning https://www.youtube.com/watch?v=D0efHEJsfHo
    """

    def __init__(
        self,
        _min_samples_split,
        _min_impurity,
        _max_depth,
        _loss=None,
    ):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = _min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = _min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = _max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # function ti post prune the tree
        self._post_prune = None
        # For Gradient Boosting
        self.loss = _loss
        # Number of features to choose randompy.For random Forest.
        _n_rand_features = None

    def fit(self, X, y, loss=None):
        """Build decision tree"""
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._grow_tree(X, y)
        self.loss = None

    def _grow_tree(self, X, y, current_depth=0):
        """Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""

        best_split_data = {}  # Subsets of the data

        # Check if expansion of y is needed
        # if len(np.shape(y)) == 1:
        #     print("defbg n")
        #     print(y)
        #     y = np.expand_dims(y, axis=1)
        #     print(y)

        # Add y as last column of X
        dataset = np.c_[X, y]
        # dataset = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)
        largest_impurity = -1
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_index in range(n_features):
                # All values of feature_index
                feature_values = X[:, feature_index]
                possible_thresholds = np.unique(feature_values)

                # Iterate through all unique values(possible_thresholds) of feature column i and
                # calculate the impurity
                for threshold in possible_thresholds:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold
                    dataset_left, dataset_right = split_data_by_feature(
                        dataset, feature_index, threshold
                    )

                    if len(dataset_left) > 0 and len(dataset_right) > 0:
                        # Select the y-values of the two sets (SET TO -1 n_features:)
                        left_child_labels, right_child_labels = (
                            dataset_left[:, n_features:],
                            dataset_right[:, n_features:],
                        )
                        # Calculate impurity
                        current_impurity = self._impurity_calculation(
                            y, left_child_labels, right_child_labels
                        )

                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature
                        # index
                        if current_impurity > largest_impurity:
                            best_split_data["feature_index"] = feature_index
                            best_split_data["threshold"] = threshold
                            best_split_data["leftX"] = dataset_left[:, :n_features]
                            best_split_data["leftX"] = dataset_left[
                                :, :n_features
                            ]  # X of left subtree
                            best_split_data["lefty"] = dataset_left[
                                :, n_features:
                            ]  # y of left subtree
                            best_split_data["rightX"] = dataset_right[
                                :, :n_features
                            ]  # X of right subtree
                            best_split_data["righty"] = dataset_right[
                                :, n_features:
                            ]  # y of right subtree

                            largest_impurity = current_impurity

        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            left_subtree = self._grow_tree(
                best_split_data["leftX"], best_split_data["lefty"], current_depth + 1
            )
            right_subtree = self._grow_tree(
                best_split_data["rightX"], best_split_data["righty"], current_depth + 1
            )

            # if leaf nodes have equal values
            # self._merge()

            # here post prunning can be applied for classifiation or regression
            # self._post_prune()

            return Node(
                _feature_index=best_split_data["feature_index"],
                _threshold=best_split_data["threshold"],
                _left=left_subtree,
                _right=right_subtree,
            )

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)

        return Node(_value=leaf_value)

    def _make_prediction(self, sample, node):
        """Do a recursive search down the tree and make a prediction of the data sample by the
        value of the leaf that we end up at"""

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if node._is_leaf_node():
            return node.value

        feature_value = sample[node.feature_index]
        if feature_value <= node.threshold:
            return self._make_prediction(sample, node.left)
        return self._make_prediction(sample, node.right)

    def predict(self, X):
        """Classify samples one by one and return the set of labels"""

        return [self._make_prediction(sample, self.root) for sample in X]


class RegressionTree(DecisionTree):
    def __init__(
        self,
        # Minimum n of samples to justify split
        _min_samples_split=2,
        # The minimum impurity to justify split
        _min_impurity=1e-7,
        # The maximum depth to grow the tree to
        _max_depth=float("inf"),
        # Number of features to choose randompy.For random Forest.
        _n_rand_features=None,
    ):
        super(DecisionTree, self).__init__(
            _min_samples_split=_min_samples_split,
            _min_impurity=_min_impurity,
            _max_depth=_max_depth,
            _n_rand_features=_n_rand_features,
        )

    def _calculate_variance_reduction(
        self, parent_labels, left_child_labels, right_child_labels
    ):
        weight_l = len(left_child_labels) / len(parent_labels)
        weight_r = len(right_child_labels) / len(parent_labels)
        reduction = np.var(parent_labels) - (
            weight_l * np.var(left_child_labels) + weight_r * np.var(right_child_labels)
        )

        return reduction

    def _leaf_mean_value(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._leaf_mean_value
        super(RegressionTree, self).fit(X, y)


class ClassificationTree(DecisionTree):
    def __init__(
        self,
        # Minimum n of samples to justify split
        _min_samples_split=2,
        # The minimum impurity to justify split
        _min_impurity=1e-7,
        # The maximum depth to grow the tree to
        _max_depth=float("inf"),
        # Information gein calculation criterion
        _criterion="entropy",
    ):
        super(DecisionTree, self).__init__(
            _min_samples_split=_min_samples_split,
            _min_impurity=_min_impurity,
            _max_depth=_max_depth,
        )
        self.criterion = _criterion

    def _calculate_information_gain(
        self, parent_labels, left_child_labels, right_child_labels, criterion
    ):
        weight_l = len(left_child_labels) / len(parent_labels)
        weight_r = len(right_child_labels) / len(parent_labels)

        if criterion == "gini":
            return calculate_gini_index(parent_labels) - (
                weight_l * calculate_gini_index(left_child_labels)
                + weight_r * calculate_gini_index(right_child_labels)
            )
        elif criterion == "entropy":
            return calculate_entropy(parent_labels) - (
                weight_l * calculate_entropy(left_child_labels)
                + weight_r * calculate_entropy(right_child_labels)
            )
        elif criterion == "error_rate":
            return calculate_error_rate(parent_labels) - (
                weight_l * calculate_error_rate(left_child_labels)
                + weight_r * calculate_error_rate(right_child_labels)
            )
        else:
            return calculate_laplace_error_rate(parent_labels) - (
                weight_l * calculate_laplace_error_rate(left_child_labels)
                + weight_r * calculate_laplace_error_rate(right_child_labels)
            )

    def _majority_vote(self, y):
        y = list(y)
        return max(y, key=y.count)

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        self._post_prune = self._cost_complexity_pruning
        super(ClassificationTree, self).fit(X, y)

    # post prunning

    def _cost_complexity_pruning(self, tree):
        return tree

    def _rule_post_pruning(self, tree):
        return tree

    def _reduced_error_pruning(self, tree):
        return tree

    # def _pessimistic_error_pruning(self):
    #     self._calculate_error(self.root)

    #     # LIFO processing
    #     stack = []
    #     stack.append(self.root)
    #     while True:
    #         if len(stack):
    #             pop_node = stack.pop()
    #             if pop_node.left:
    #                 if pop_node.backuperror > pop_node.miss_class_probability:
    #                     pop_node = None
    #                 else:
    #                     stack.append(pop_node.right)
    #                     stack.append(pop_node.left)
    #         else:
    #             break

    # def _calculate_error(self, node):
    #     # Misclassification probability using Laplace's Law

    #     if (
    #         node.left
    #     ):  # There are child nodes, the backuperror of this node is the weighted sum of the backuperrors of sons
    #         backuperror_left = self._calculate_error(node.left)
    #         backuperror_right = self._calculate_error(node.right)

    #         node.backuperror = (
    #             len(node.left.data) / len(node.data) * backuperror_left
    #             + len(node.right.data) / len(node.data) * backuperror_right
    #         )

    #         node.miss_class_probability = Base._laplace_error_rate(node.data.to_numpy())
    #     else:  # No son nodes, backuperror = mcp
    #         node.backuperror = node.miss_class_probability = Base._laplace_error_rate(
    #             node.data.to_numpy()
    #         )
    #     return node.backuperror


class XGBoostRegressionTree(DecisionTree):
    """
    Regression tree for XGBoost
    - Reference -
    http://xgboost.readthedocs.io/en/latest/model.html
    """

    def _split(self, y):
        """y contains y_true in left half of the middle column and
        y_pred in the right half. Split and return the two matrices"""
        col = int(np.shape(y)[1] / 2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred):
        nominator = np.power((y * self.loss.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)

    def _gain_by_taylor(self, y, y1, y2):
        # Split
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)

        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _approximate_update(self, y):
        # y split into y, y_pred
        y, y_pred = self._split(y)
        # Newton's Method
        gradient = np.sum(y * self.loss.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation = gradient / hessian

        return update_approximation

    def fit(self, X, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)
