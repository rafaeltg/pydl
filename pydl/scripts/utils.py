import tensorflow as tf
from ..models.utils import load_json
from ..hyperopt.components import hp_feature_selection, hp_space_from_json


def init_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def get_search_space(s: str, features: list = None):
    space = load_json(s)

    if features is not None:
        space['config']['features'] = hp_feature_selection(len(features)).to_json()

    return hp_space_from_json(space)
