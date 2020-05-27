import os
import sys
import json
import inspect
import keras.models as k_models
from keras.utils.io_utils import H5Dict


__all__ = ['expand_arg',
           'load_model',
           'model_from_config',
           'model_from_json',
           'load_json',
           'get_custom_objects',
           'valid_act_functions',
           'valid_loss_functions',
           'valid_optimizers',
           'check_filepath']


valid_act_functions = ['softmax',
                       'softplus',
                       'sigmoid',
                       'tanh',
                       'relu',
                       'linear']

valid_loss_functions = ['mse',                       # Mean Squared Error
                        'mae',                       # Mean Absolute Error
                        'mape',                      # Mean Absolute Percentage Error
                        'msle',                      # Mean Squared Logarithmic Error
                        'binary_crossentropy',       # Log loss
                        'categorical_crossentropy',  # Multiclass Log loss
                        'kld'                        # Kullback Leibler Divergence (information gain)
                        ]

valid_optimizers = ['sgd',
                    'rmsprop',
                    'adagrad',
                    'adadelta',
                    'adam',
                    'adamax',
                    'nadam',
                    'ftrl']


def expand_arg(layers, arg_to_expand):

    """Expands the arg_to_expand into the length of layers.
    This is used as a convenience so that the user does not need to specify the
    complete list of parameters for model initialization.
    IE: the user can just specify one parameter and this function will expand it
    :param layers:
    :param arg_to_expand:
    :return:
    """

    if not isinstance(arg_to_expand, list):
        arg_to_expand = [arg_to_expand]

    if len(arg_to_expand) == len(layers):
        return arg_to_expand

    if len(arg_to_expand) > len(layers):
        return arg_to_expand[0:len(layers)]

    missing_values = len(layers) - len(arg_to_expand)
    result = arg_to_expand + [arg_to_expand[-1] for _ in range(missing_values)]

    return result


def get_custom_objects(c_objs: dict = None) -> dict:
    custom_objects = dict(inspect.getmembers(sys.modules['pydl.models'], inspect.isclass))
    if c_objs is not None:
        if isinstance(c_objs, dict):
            custom_objects.update(c_objs)
        else:
            raise ValueError('Invalid custom_objects')

    return custom_objects


def load_model(filepath, custom_objects: dict = None, compile: bool = True):
    if isinstance(filepath, str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError('{} does not exists.'.format(filepath))

        if not filepath.endswith('.h5'):
            raise ValueError('Input file must be in H5 format.')

    try:
        c_objs = get_custom_objects(custom_objects)
        model = k_models.load_model(filepath, custom_objects=c_objs, compile=compile)
    except:
        try:
            model = load_pipeline(filepath, custom_objects=custom_objects, compile=compile)
        except:
            model = load_base_model(filepath, custom_objects=custom_objects)

    return model


def load_pipeline(filepath, custom_objects: dict = None, compile: bool = False):
    with H5Dict(filepath, mode='r') as h5dict:
        model_config = h5dict['model_config']
        if model_config is None:
            raise ValueError('No model found in config.')

        c = {}

        config = model_config['config']

        name = config.get('name', None)
        if name is not None:
            c['name'] = name

        steps = config.get('steps', None)
        c['steps'] = [model_from_json(s.decode('utf-8'), custom_objects=custom_objects) for s in steps]

        estimator = config.get('estimator', None)
        if estimator is None:
            raise ValueError('No estimator found in config.')

        c['steps'].append(load_model(estimator.data, custom_objects=custom_objects, compile=compile))

    return get_custom_objects()['Pipeline'](**c)


def load_base_model(filepath, custom_objects: dict = None):
    with H5Dict(filepath, mode='r') as h5dict:
        model_config = h5dict.get('model_config', None)
        if model_config is None:
            raise ValueError('No model found in config.')

        model_config = json.loads(model_config.decode('utf-8'))
        model = model_from_config(model_config, custom_objects=custom_objects)

    return model


def model_from_config(config: dict, custom_objects: dict = None):
    c_objs = get_custom_objects(custom_objects)
    return k_models.model_from_config(config, c_objs)


def model_from_json(model_json: str,
                    custom_objects: dict = None,
                    weights_filepath: str = None,
                    compile: bool = False):

    if model_json.endswith('.json') and os.path.isfile(model_json):
        with open(model_json) as json_file:
            model_json = json_file.read()

    c_objs = get_custom_objects(custom_objects)
    model = k_models.model_from_json(model_json, c_objs)

    if weights_filepath:
        model.load_weights(weights_filepath)

    if compile:
        model.compile(
            optimizer=model.get_optimizer(),
            loss=model.get_loss_func()
        )

    return model


def load_json(json_string: str):
    if os.path.isfile(json_string):
        with open(json_string) as file:
            data = json.load(file)
    else:
        data = json.loads(json_string)

    return data


def check_filepath(filepath: str, name: str, extension: str):
    if filepath is None:
        filepath = os.getcwd()

    ext = '.' + extension

    if not filepath.endswith(ext):
        filepath = os.path.join(filepath, '{}{}'.format(name, ext))

    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return filepath