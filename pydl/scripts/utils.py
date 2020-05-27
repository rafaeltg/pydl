import tensorflow as tf
from ..models.utils import load_json
from ..models.pipeline import hp_base_filter_select
from ..hyperopt import hp_space_from_json


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
        if space['class_name'] == 'Pipeline':
            space['config']['steps']['config']['value'] = [
                hp_base_filter_select(n_features=len(features)).to_json()
            ] + space['config']['steps']['config']['value']
        else:
            space = dict(
                class_name='Pipeline',
                config=dict(
                    steps=[
                        hp_base_filter_select(n_features=len(features)).to_json(),
                        space
                    ]
                )
            )

    return hp_space_from_json(space)
