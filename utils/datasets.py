import collections
import csv
import numpy as np
import os
import tempfile
from tensorflow.python.platform import gfile

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'test', 'validation'])


def load_csv(filename, data_type=np.float64, has_header=True):
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)

        if has_header:
            header = next(data_file)

        data = []
        for ir in data_file:
            data.append(np.asarray(ir, dtype=data_type))

        data = np.asarray(data, dtype=data_type)

    return data


def load_npy(filename):
    return np.load(filename) if filename != '' else None


def load_dataset(x_path, y_path='', y_dtype=np.float64, has_header=True):

    if x_path == '':
        return Dataset(data=None, target=None)

    if os.path.splitext(x_path)[1] == '.csv':
        x = load_csv(filename=x_path, has_header=has_header)

    else:
        x = load_npy(x_path)

    if y_path is '':
        y = None

    elif os.path.splitext(y_path)[1] == '.csv':
        y = load_csv(filename=y_path, data_type=y_dtype, has_header=has_header)

    else:
        y = load_npy(y_path)

    return Dataset(data=x, target=y)


def load_datasets(train_dataset,
                  train_labels='',
                  test_dataset='',
                  test_labels='',
                  valid_dataset='',
                  valid_labels='',
                  labels_dtype=np.float64,
                  has_header=True):

    """
    """

    assert train_dataset != ''

    return Datasets(train=load_dataset(train_dataset, train_labels, labels_dtype, has_header),
                    test=load_dataset(test_dataset, test_labels, labels_dtype, has_header),
                    validation=load_dataset(valid_dataset, valid_labels, labels_dtype, has_header))


def maybe_download(filename, work_directory, source_url):

    """Download the data from source url, unless it's already here.
    :param filename: string, name of the file in the directory.
    :param work_directory: string, path to working directory.
    :param source_url: url to download from if file doesn't exist.
    :return: path to resulting file.
    """

    if not gfile.Exists(work_directory):
        gfile.MakeDirs(work_directory)
        filepath = os.path.join(work_directory, filename)

    if not gfile.Exists(filepath):
        with tempfile.NamedTemporaryFile() as tmpfile:
            temp_file_name = tmpfile.name
            urllib.request.urlretrieve(source_url, temp_file_name)
            gfile.Copy(temp_file_name, filepath)

        with gfile.GFile(filepath) as f:
            size = f.Size()
        print('Successfully downloaded', filename, size, 'bytes.')

    return filepath
