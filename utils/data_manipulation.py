import numpy as np



def shuffle_data(X, y, seed=None):
    """Random shuffle of the samples in X and y"""
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def batch_iterator(X, y=None, batch_size=64):
    """Simple batch generator"""
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


def split_data_by_feature(dataset, feature_index, threshold):
    """Divide dataset based on if sample value on feature index is larger than
    the given threshold

    Binarisation (partition logic) can be modified
    """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        print("split", "isinstance")
        split_func = lambda sample: sample[feature_index] >= threshold
    else:
        print("split", "is not instance")
        split_func = lambda sample: sample[feature_index] == threshold

    dataset_left = np.array([sample for sample in dataset if split_func(sample)])
    dataset_right = np.array([sample for sample in dataset if not split_func(sample)])

    return np.array([dataset_left, dataset_right])


def bootstrap_samples(X, y):
    """Return random subsets with replacements of the data"""
    n_samples = X.shape[0]
    random_indexes = np.random.choice(n_samples, n_samples, replace=True)
    return X[random_indexes], y[random_indexes]


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """Split the data into train and test sets"""
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def k_fold_cross_validation_sets(X, y, k, shuffle=True):
    """Split the data into k sets of training / test data"""
    if shuffle:
        X, y = shuffle_data(X, y)

    n_samples = len(y)
    left_overs = {}
    n_left_overs = n_samples % k
    if n_left_overs != 0:
        left_overs["X"] = X[-n_left_overs:]
        left_overs["y"] = y[-n_left_overs:]
        X = X[:-n_left_overs]
        y = y[:-n_left_overs]

    X_split = np.split(X, k)
    y_split = np.split(y, k)
    sets = []
    for i in range(k):
        X_test, y_test = X_split[i], y_split[i]
        X_train = np.concatenate(X_split[:i] + X_split[i + 1 :], axis=0)
        y_train = np.concatenate(y_split[:i] + y_split[i + 1 :], axis=0)
        sets.append([X_train, X_test, y_train, y_test])

    # Add left over samples to last set as training samples
    if n_left_overs != 0:
        np.append(sets[-1][0], left_overs["X"], axis=0)
        np.append(sets[-1][2], left_overs["y"], axis=0)

    return np.array(sets)
