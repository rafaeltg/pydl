import os
import numpy as np
from .utils import get_model, get_input_data, load_data
from pydl.models import SupervisedModel, load_model


def predict(config, output):
    """
    """

    m = load_model(get_model(config))
    assert isinstance(m, SupervisedModel), 'The given model cannot perform predict operation!'

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')

    preds = m.predict(x)

    # Save predictions as .npy file
    np.save(os.path.join(output, m.name+'_preds.npy'), preds)
