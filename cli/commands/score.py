import os
from .utils import get_input_data, load_data, get_model
from pydl.models.utils import save_json, load_model


def score(config, output):
    """
    """

    m = load_model(get_model(config))

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')
    y = load_data(data_set, 'data_y') if 'data_y' in data_set else None

    if y is None:
        s = m.score(x)
    else:
        s = m.score(x, y)

    result = {m.get_loss_func(): s}

    save_json(result, os.path.join(output, m.name+'_score.json'))
