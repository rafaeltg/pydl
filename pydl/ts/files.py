import os
import pandas
import numpy as np


__all__ = ['load_csv', 'load_npy', 'load_data_file']


def load_csv(filename, dtype=None, has_header=True, usecols=None, index_col=None):
    return pandas.read_csv(filename,
                           skiprows=0 if has_header else None,
                           header=0 if has_header else None,
                           index_col=index_col,
                           usecols=usecols,
                           dtype=dtype,
                           parse_dates=True)


def load_npy(filename):
    return np.load(filename) if filename != '' else None


def load_data_file(filename, **kwargs):
    if filename is '':
        return None
    elif os.path.splitext(filename)[1] == '.csv':
        return load_csv(filename=filename, **kwargs)
    else:
        return load_npy(filename)
