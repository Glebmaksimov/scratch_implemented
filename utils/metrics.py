import numpy as np


# COMMON


def calculate_variance(X):
    """Return the variance of the features in dataset X"""
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]

    return (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))


def calculate_std_dev(X):
    """Calculate the standard deviations of the features in dataset X"""
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev


def calculate_covariance_matrix(X, Y=None):
    """Calculate the covariance matrix for the dataset X"""
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(
        Y - Y.mean(axis=0)
    )

    return np.array(covariance_matrix, dtype=float)


def calculate_correlation_matrix(X, Y=None):
    """Calculate the correlation matrix for the dataset X"""
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)


# REGRESSION ACCURACY METRICS


def mean_squared_error(y_true, y_pred):
    """Returns the mean squared error between y_true and y_pred"""
    return np.mean(np.power(y_true - y_pred, 2))


def SSR(y, y_pred):
    return sum((y_pred - np.mean(y)) ** 2)


def SSE(y, y_pred):
    return sum((y_pred - y) ** 2)


def SST(y, y_pred):
    return SSR(y, y_pred) + SSE(y, y_pred)


def mean_absolute_error(y, y_pred):
    return np.mean(abs(y - y_pred))


def root_mean_squared_error(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def residual_standart_error(y, y_pred, p):
    return np.sqrt((SSE(y, y_pred) / (len(y) + p + 1)))


def r_squared(y, y_pred):
    return SSR(y, y_pred) / SST(y, y_pred)


def accuracy_score(y_true, y_pred):
    """Compare y_true to y_pred and return the accuracy"""
    return np.sum(y_true == y_pred) / len(y_true)


# CLASSIFICATION METRICS


def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n = len(classes)
    confusion_matrics = np.zeros(shape=(n, n), dtype=np.int32)
    for i, j in zip(y_true, y_pred):
        confusion_matrics[
            np.where(classes == i)[0], np.where(classes == j)[0]
        ] += 1  # noqa

    return confusion_matrics



def calculate_entropy(y):
    """Calculates the entropy of label array y"""

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()

    return sum(-probabilities * np.log2(probabilities))


def calculate_gini_index(y):
    """Calculates the gini index of label array y"""

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()

    return 1 - sum(np.square(probabilities))

def calculate_error_rate(y):
    """Calculates the error rate of label array y"""

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()

    return 1 - np.max(probabilities)


def calculate_laplace_error_rate(y):
    """Calculates the laplace error rate of label array y"""

    _, counts = np.unique(y, return_counts=True)

    c = np.max(counts)
    k = counts.sum()

    return (k - c + 1) / (k + 2)


def precision_score(y_true, y_pred, average="auto"):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    if average == "auto":
        if len(classes) == 2:
            average = "binary"
        else:
            average = "micro"

    cm = confusion_matrix(y_true, y_pred)
    if average == "binary":
        tp, fp = cm.ravel()[:2]
        return tp / (tp + fp)

    if average == "micro":
        tp, fp = list(), list()
        for i in range(len(cm)):
            tp.append(cm[i, i])
            fp.append(sum(np.delete(cm[i], i)))

        tp_all = sum(tp)
        fp_all = sum(fp)
        return tp_all / (tp_all + fp_all)


def recall_score(y_true, y_pred, average="auto"):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    if average == "auto":
        if len(classes) == 2:
            average = "binary"
        else:
            average = "micro"

    cm = confusion_matrix(y_true, y_pred)
    if average == "binary":
        tp, fn = cm.ravel()[[0, 2]]
        return tp / (tp + fn)

    if average == "micro":
        tp, fn = list(), list()
        for i in range(len(cm)):
            tp.append(cm[i, i])
            fn.append(sum(np.delete(cm[:, i], i)))

        tp_all = sum(tp)
        fn_all = sum(fn)
        return tp_all / (tp_all + fn_all)


# DISTANCE ACCURACY METRICS


def euclidian_distance(x_1, x_2):
    """Calculates the l2 distance between two vectors"""
    return np.linalg.norm(x_1 - x_2, ord=2)


def manghattan_distance(x_1, x_2):
    """Calculates the Manghattan between two vectors"""
    return np.linalg.norm(x_1 - x_2, ord=1)


def minkowski_distance(x_1, x_2, power):
    """Calculates the Minkowski between two vectors"""
    return np.linalg.norm(x_1 - x_2, ord=power)


def hamming_distance(x_1, x_2):
    """Calculates the Hamming between two vectors"""
    return [x_1[i] != x_2[i] for i in range(len(x_2))]


def cosine_distance(x_1, x_2):
    """Calculates the Cosine between two vectors"""
    return np.dot(x_1, x_2) / np.sqrt(np.dot(x_1) * np.dot(x_2))
