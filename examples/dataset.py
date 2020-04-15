import numpy as np
from pydl.ts.transform import train_test_split, split_sequences


def create_univariate_data(size=200):
    return np.random.random_sample((size, 1)) + 1


def create_multivariate_data(size=200, n_features=2) -> np.ndarray:
    data = []
    # define multivariate input sequence
    for i in range(n_features):
        data.append(np.random.random_sample((size, 1)) + 1)

    data.append(np.sum(data, axis=0))

    # horizontally stack columns
    return np.hstack(data)


def create_multivariate_dataset(size=200, n_steps=3, n_features=2, train_size=0.8):

    dataset = create_multivariate_data(size, n_features)

    # convert into input/output sequences
    x, y = split_sequences(dataset[:, :-1], dataset[:, -1], n_steps)
    y = np.reshape(y, (len(y), 1))

    return train_test_split(x, y, train_size)
