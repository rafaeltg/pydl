import collections
import os

import numpy as np
import pandas

Dataset = collections.namedtuple('Dataset', ['x', 'y'])
Datasets = collections.namedtuple('Datasets', ['train', 'test', 'validation'])


def load_csv(filename, dtype=np.float64, has_header=True, usecols=None):
    return pandas.read_csv(filename, skiprows=0 if has_header else None, usecols=usecols).values.astype(dtype)


def load_npy(filename):
    return np.load(filename) if filename != '' else None


def load_data_file(filename, dtype=np.float64, has_header=True):
    if filename is '':
        return None
    elif os.path.splitext(filename)[1] == '.csv':
        return load_csv(filename=filename, dtype=dtype, has_header=has_header)
    else:
        return load_npy(filename)


def load_dataset(x_path, y_path='', y_dtype=np.float64, has_header=True):
    x = load_data_file(filename=x_path, has_header=has_header)
    y = load_data_file(filename=y_path, dtype=y_dtype, has_header=has_header)
    return Dataset(x=x, y=y)


def load_datasets(train_x,
                  train_y='',
                  test_x='',
                  test_y='',
                  valid_x='',
                  valid_y='',
                  labels_dtype=np.float64,
                  has_header=True):

    """
    """

    assert train_x != ''

    return Datasets(train=load_dataset(train_x, train_y, labels_dtype, has_header),
                    test=load_dataset(test_x, test_y, labels_dtype, has_header),
                    validation=load_dataset(valid_x, valid_y, labels_dtype, has_header))


def create_dataset(dataset, look_back=1, time_ahead=1):
    """

    :param dataset:
    :param look_back:
    :param time_ahead:
    :return:
    """

    assert len(dataset) > look_back+time_ahead, 'Dataset too small!'
    y_starts = range(look_back, len(dataset)+1-time_ahead, 1)
    y_idxs = [range(i, i+time_ahead) for i in y_starts]
    x_idxs = [range(i-look_back, i) for i in y_starts]
    data_x, data_y = [], []
    for i in range(len(y_idxs)):
        data_x.append(dataset[x_idxs[i], 0])
        data_y.append(np.reshape(dataset[y_idxs[i], 0], time_ahead))
    return np.array(data_x), np.array(data_y)


def get_stock_historical_data(symbol, start, end, ascending=True, usecols=None):

    """
    :param symbol: stock ticker.
    :param start: string date in format 'yyyy-mm-dd' ('2009-09-11').
    :param end: string date in format 'yyyy-mm-dd' ('2010-09-11').
    :param ascending: sort returning values in ascending or descending order based on Date column.
    :param usecols: List of columns to return. If None, return all columns.
    :return: DataFrame
    """

    from yahoo_finance import Share

    stock = Share(symbol)
    data = stock.get_historical(start, end)

    df = pandas.DataFrame(data).sort_values(by='Date', ascending=ascending)
    df = df.drop('Symbol', 1)
    df['Date'] = pandas.to_datetime(df['Date'], format='%Y-%m-%d')
    for c in df.columns.difference(['Date']):
        df[c] = pandas.to_numeric(df[c])
    return df if usecols is None else df[usecols]


def get_return(x):
    if not isinstance(x, pandas.DataFrame):
        x = pandas.DataFrame(x)
    return x.pct_change(1) + 1


def get_log_return(x):
    if not isinstance(x, pandas.DataFrame):
        x = pandas.DataFrame(x)
    return np.log(x).diff(1)[1:].as_matrix()
