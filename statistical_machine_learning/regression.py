import numpy as np
from scipy.stats import chi2, multivariate_normal
from scratch_implemented.utils import normalize, polynomial_features


# COMMON CONSTANTS

LEARNING_RATE = 0.01
TOLERANCE = 0.0001
ITERARIONS = 1000
DEGREE = None
ALPHA = 0.1
GRADIENT_DESCENT = True
L1_RATIO = 0.5


class l1_regularization:
    """Regularization for Lasso Regression"""

    def __init__(self, _alpha):
        self.alpha = _alpha

    # this will allow you to treat object as function
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)


class l2_regularization:
    """Regularization for Ridge Regression"""

    def __init__(self, _alpha):
        self.alpha = _alpha

    def __call__(self, w):
        # we need 0.5 to get rid of 2 after derivation
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w


class l1_l2_regularization:
    """Regularization for Elastic Net Regression"""

    def __init__(self, _alpha, _l1_ratio):
        self.alpha = _alpha
        self.l1_ratio = _l1_ratio

    def __call__(self, w):
        l1_term = self.l1_ratio * np.linalg.norm(w)
        l2_term = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_term + l2_term)

    def grad(self, w):
        l1_term = self.l1_ratio * np.sign(w)
        l2_term = (1 - self.l1_ratio) * w
        return self.alpha * (l1_term + l2_term)


class Regression(object):
    """Base regression model. Models the relationship between a scalar dependent variable y and the independent
    variables X.

    Parameters:
    -----------
    _n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    _learning_rate: float
        The step length that will be used when updating the weights.

    -----------
    References:
    """

    def __init__(self, _n_iterations, _learning_rate, _tolerance):
        self.n_iterations = _n_iterations
        self.learning_rate = _learning_rate
        self.tolerance = _tolerance

    def fit(self, X, y):
        # Whenever you have a convex cost function you are allowed to initialize your weights to zeros.
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0

        self.training_errors = []
        self.training_errors.append(np.inf)

        # Do gradient descent for n_iterations
        for i in range(1, self.n_iterations):
            y_pred = np.dot(X, self.weights)
            # Calculate l2 loss(MSE)
            mse = np.mean(0.5 * (y - y_pred) ** 2 + self.regularization(self.weights))
            self.training_errors.append(mse)
            # Gradient of l2 loss w.r.t w
            d_weights = (
                2
                / n_samples
                * (np.dot(y_pred - y, X) + self.regularization.grad(self.weights))
            )
            d_intercept = 2 / n_samples * np.sum(y_pred - y)
            # Update the weights
            self.weights -= self.learning_rate * d_weights
            self.intercept -= self.learning_rate * d_intercept

            if (
                abs(self.training_errors[i] - self.training_errors[i - 1])
                < self.tolerance
            ):
                break

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept


class LinearRegressor(Regression):
    """Linear model.

    Parameters:
    -----------
    _n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    _learning_rate: float
        The step length that will be used when updating the weights.
    _gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If
        false then we use batch optimization by least squares.
    -----------

    """

    def __init__(
        self,
        _n_iterations=ITERARIONS,
        _learning_rate=LEARNING_RATE,
        _tolerance=TOLERANCE,
        _gradient_descent=GRADIENT_DESCENT,
    ):
        self.gradient_descent = _gradient_descent
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

        super(LinearRegressor, self).__init__(
            _n_iterations=_n_iterations,
            _learning_rate=_learning_rate,
            _tolerance=_tolerance,
        )

    def fit(self, X, y):
        # If not gradient descent => Least squares approximation of w
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.weights = X_sq_reg_inv.dot(X.T).dot(y)

            # enother way to perform ols(Least squares approximation of w) - better TODO

            # ones = np.ones(len(X)).reshape(-1, 1)
            # X = np.concatenate((ones, X), axis=1)

            # model_coeficients = np.matmul(
            #     np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, y)
            # )  # noqa

            # self.weights = model_coeficients[1:]
            # self.intercept = model_coeficients[0]

        else:
            super(LinearRegressor, self).fit(X, y)


class LassoRegressor(Regression):
    """Linear regression model with a regularization factor which does both variable selection
    and regularization. Model that tries to balance the fit of the model with respect to the training
    data and the complexity of the model. A large regularization factor with decreases the variance of
    the model and do para.

    Parameters:
    -----------

    _degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    _alpha: float
        The factor that will determine the amount of regularization and feature
        shrinkage.
    _n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    _learning_rate: float
        The step length that will be used when updating the weights.

    -----------

    """

    def __init__(
        self,
        _degree=DEGREE,
        _alpha=ALPHA,
        _n_iterations=ITERARIONS,
        _learning_rate=LEARNING_RATE,
        _tolerance=TOLERANCE,
    ):
        self.degree = _degree
        self.regularization = l1_regularization(_alpha=_alpha)
        super(LassoRegressor, self).__init__(
            _n_iterations=_n_iterations,
            _learning_rate=_learning_rate,
            _tolerance=_tolerance,
        )

    def fit(self, X, y):
        if self.degree is not None:
            X = normalize(polynomial_features(X, _degree=self.degree))
        super(LassoRegressor, self).fit(X, y)

    def predict(self, X):
        if self.degree is not None:
            X = normalize(polynomial_features(X, _degree=self.degree))
        return super(LassoRegressor, self).predict(X)


class RidgeRegressor(Regression):
    """Also referred to as Tikhonov regularization. Linear regression model with a regularization factor.
    Model that tries to balance the fit of the model with respect to the training data and the complexity
    of the model. A large regularization factor with decreases the variance of the model.

    Parameters:
    -----------
    _degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    _alpha: float
        The factor that will determine the amount of regularization and feature
        shrinkage.
    _n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    _learning_rate: float
        The step length that will be used when updating the weights.

    """

    def __init__(
        self,
        _degree=DEGREE,
        _alpha=ALPHA,
        _n_iterations=ITERARIONS,
        _learning_rate=LEARNING_RATE,
        _tolerance=TOLERANCE,
    ):
        self.degree = _degree
        self.regularization = l2_regularization(_alpha=_alpha)
        super(RidgeRegressor, self).__init__(
            _n_iterations=_n_iterations,
            _learning_rate=_learning_rate,
            _tolerance=_tolerance,
        )

    def fit(self, X, y):
        if self.degree is not None:
            X = normalize(polynomial_features(X, _degree=self.degree))
        super(RidgeRegressor, self).fit(X, y)

    def predict(self, X):
        if self.degree is not None:
            X = normalize(polynomial_features(X, _degree=self.degree))
        return super(RidgeRegressor, self).predict(X)


class ElasticNetRegressor(Regression):

    """Regression where a combination of l1 and l2 regularization are used. The
    ratio of their contributions are set with the 'l1_ratio' parameter.

    Parameters:
    -----------
    _degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    _alpha: float
        The factor that will determine the amount of regularization and feature
        shrinkage.
    _l1_ration: float
        Weighs the contribution of l1 and l2 regularization.
    _n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    _learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(
        self,
        _degree=DEGREE,
        _alpha=ALPHA,
        _l1_ratio=L1_RATIO,
        _tolerance=TOLERANCE,
        _n_iterations=ITERARIONS,
        _learning_rate=LEARNING_RATE,
    ):
        self.degree = _degree
        self.regularization = l1_l2_regularization(_alpha=_alpha, _l1_ratio=_l1_ratio)
        super(ElasticNetRegressor, self).__init__(
            _n_iterations=_n_iterations,
            _learning_rate=_learning_rate,
            _tolerance=_tolerance,
        )

    def fit(self, X, y):
        if self.degree is not None:
            X = normalize(polynomial_features(X, _degree=self.degree))
        super(ElasticNetRegressor, self).fit(X, y)

    def predict(self, X):
        if self.degree is not None:
            X = normalize(polynomial_features(X, _degree=self.degree))
        return super(ElasticNetRegressor, self).predict(X)


class PolynomialRegressor(Regression):
    """Performs a non-linear transformation of the data before fitting the model
    and doing predictions which allows for doing non-linear regression.

    Parameters:
    -----------

    _degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    _n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    _learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(
        self,
        _degree=2 if DEGREE is None else DEGREE,
        _n_iterations=ITERARIONS,
        _learning_rate=LEARNING_RATE,
        _tolerance=TOLERANCE,
    ):
        self.degree = _degree
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(PolynomialRegressor, self).__init__(
            _n_iterations=_n_iterations,
            _learning_rate=_learning_rate,
            _tolerance=_tolerance,
        )

    def fit(self, X, y):
        X = polynomial_features(X, _degree=self.degree)
        super(PolynomialRegressor, self).fit(X, y)

    def predict(self, X):
        X = polynomial_features(X, _degree=self.degree)
        return super(PolynomialRegressor, self).predict(X)
