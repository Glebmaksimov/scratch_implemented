import numpy as np
import math
from scratch_implemented.utils import make_diagonal
from scratch_implemented.deep_learning.activations import Sigmoid


class LogisticRegression:
    """Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If
        false then we use batch optimization by least squares.

    # TODO

     - Make it multiclass using SoftMax

    """

    def __init__(self, learning_rate=0.1, gradient_descent=True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for _ in range(n_iterations):
            # Make a new prediction
            y_pred = self.sigmoid(X.dot(self.param))
            if self.gradient_descent:
                # Move against the gradient of the loss function with
                # respect to the parameters to minimize the loss
                self.param -= self.learning_rate * -(y - y_pred).dot(X)
            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                # Batch opt:
                self.param = (
                    np.linalg.pinv(X.T.dot(diag_gradient).dot(X))
                    .dot(X.T)
                    .dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)
                )

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred


# class Logistic_Regression_Classifier:
#     def __init__(self, learning_rate=0.0001, _tolerance=1e-4, _iterations=5000):
#         self.iterations = _iterations
#         self.learning_rate = learning_rate
#         self.tolerance = _tolerance

#     @staticmethod
#     def loss(y_true, y_pred):
#         y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
#         return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#     def sigmoid(self, X):
#         return 1 / (1 + np.exp(-X))

#     def fit(self, X, y):
#         n = X.shape[1]
#         self.weights = np.zeros(n)
#         self.intercept = 0
#         prev_loss = np.inf

#         for _ in range(self.iterations):
#             y_pred = self.sigmoid(np.dot(X, self.weights) + self.intercept)

#             d_weights = (1 / n) * np.dot(X.T, (y_pred - y))
#             d_intercept = (1 / n) * np.sum(y_pred - y)

#             self.weights -= self.learning_rate * d_weights
#             self.intercept -= self.learning_rate * d_intercept

#             current_loss = self.loss(y, y_pred)
#             if abs(prev_loss - current_loss) < self.tolerance:
#                 break

#             prev_loss = current_loss

#     def predict(self, X):
#         predictions = self.sigmoid(np.dot(X, self.weights) + self.intercept)
#         return np.where(predictions >= 0.5, 1, 0)
