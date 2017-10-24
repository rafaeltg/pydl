import os
import numpy as np
from .utils import get_model, get_input_data, load_data
from pydl.models import UnsupervisedModel, load_model


def transform(config, output):
    """
    """

    m = load_model(get_model(config))
    assert isinstance(m, UnsupervisedModel), 'The given model cannot perform transform operation!'

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')

    x_encoded = m.transform(x)

    # Save encoded data in a .npy file
    base_name = os.path.splitext(os.path.basename(data_set['data_x']))[0]
    np.save(os.path.join(output, base_name+'_encoded.npy'), x_encoded)
