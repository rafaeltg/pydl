import os

import numpy as np


valid_act_functions = ['softmax', 'softplus', 'sigmoid', 'tanh', 'relu', 'linear']

valid_loss_functions = ['mse',                       # Mean Squared Error
                        'mae',                       # Mean Absolute Error
                        'mape',                      # Mean Absolute Percentage Error
                        'msle',                      # Mean Squared Logarithmic Error
                        'binary_crossentropy',       # Log loss
                        'categorical_crossentropy',  # Multiclass Log loss
                        'kld',                       # Kullback Leibler Divergence (information gain)
                        'custom']

valid_optimization_functions = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam']


# ################ #
#   Data helpers   #
# ################ #

def gen_batches(data, batch_size):

    """ Divide input data into batches.
    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]


def normalize(data):

    """ Normalize the data to be in the [0, 1] range.
    :param data:
    :return: normalized data
    """

    out_data = data.copy()

    for i, sample in enumerate(out_data):
        out_data[i] /= sum(out_data[i])

    return out_data


# ############# #
#   Utilities   #
# ############# #

def create_dir(dir_path):

    """
    :param dir_path: directory to be created
    """

    try:
        if not os.path.exists(dir_path):
            print('Creating %s directory.' % dir_path)
            os.makedirs(dir_path)
    except OSError as e:
        raise e


def expand_args(layers, args_to_expand):

    """Expands all the lists in args_to_expand into the length of layers.
    This is used as a convenience so that the user does not need to specify the
    complete list of parameters for model initialization.
    IE: the user can just specify one parameter and this function will expand it
    :param layers:
    :param args_to_expand:
    :return:
    """

    for key, val in args_to_expand.items():
        if isinstance(val, list) and (len(val) != len(layers)):
            args_to_expand[key] = [val[0] for _ in layers]

    return args_to_expand


def flag_to_list(flag_val, dtype):

    """
    :param flag_val:
    :param dtype:
    :return:
    """

    if dtype == 'int':
        return [int(_) for _ in flag_val.split(',') if _]

    elif dtype == 'float':
        return [float(_) for _ in flag_val.split(',') if _]

    elif dtype == 'str':
        return [_ for _ in flag_val.split(',') if _]

    else:
        raise Exception("Incorrect data type")

