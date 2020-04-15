import pandas as pd
import numpy as np


__all__ = ['train_test_split',
           'series_to_supervised',
           'split_sequences',
           'diff',
           'ret',
           'log_ret',
           'smooth']


def train_test_split(x, y, train_size=0.8):
    # split into train and test sets
    test_idx = int(len(x) * train_size)
    x_train, y_train = x[:test_idx], y[:test_idx]
    x_test, y_test = x[test_idx:], y[test_idx:]
    return x_train, y_train, x_test, y_test


def series_to_supervised(ts, look_back=1, time_ahead=1):

    """
    Time series to supervised data set (X, Y)
    :param ts:
    :param look_back:
    :param time_ahead:
    :return:
    """

    assert len(ts) > look_back+time_ahead, 'No enough points in time series!'

    if isinstance(ts, pd.DataFrame) or isinstance(ts, pd.Series):
        ts = ts.as_matrix()

    if len(ts.shape) == 1:
        ts = ts.reshape(-1, 1)

    y_starts = range(look_back, len(ts)+1-time_ahead, 1)
    y_idxs = [range(i, i+time_ahead) for i in y_starts]
    x_idxs = [range(i-look_back, i) for i in y_starts]
    data_x, data_y = [], []
    for i in range(len(y_idxs)):
        data_x.append(ts[x_idxs[i], 0])
        data_y.append(np.reshape(ts[y_idxs[i], 0], time_ahead))
    return np.array(data_x), np.array(data_y)


# split a multivariate sequence into samples
def split_sequences(x, y=None, n_steps=1):
    x_out, y_out = list(), list()

    for i in range(len(x)-n_steps):
        # find the end of this pattern
        end_ix = i + n_steps

        # gather input and output parts of the pattern
        x_out.append(x[i:end_ix])

        if y is not None:
            y_out.append(y[end_ix-1])

    return np.array(x_out), (np.array(y_out) if y is not None else None)


def diff(x, periods=1):

    """
    1st discrete difference

    :param x: time series
    :param periods: Periods to shift for forming difference
    :return:
    """

    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    return x.diff(periods)[periods:]


def ret(x, periods=1):

    """
    Percentage of change

    :param x:
    :param periods:
    :return:
    """
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    return x.pct_change(periods)[periods:]


def log_ret(x, periods=1):

    """
    Log percentage of change

    :param x:
    :param periods:
    :return:
    """

    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    return np.log(x).diff(periods).dropna()


def smooth(ts, method='mean', window=5):

    if method == 'mean':
        return ts.rolling(window=window).mean()

    if method == 'ewma':
        return ts.ewm(span=window).mean()

    raise Exception('Invalid method')
