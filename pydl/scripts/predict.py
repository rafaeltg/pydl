import os
import numpy as np
from ..models.utils import save_json


def predict(model,
            x,
            save_format: str = None,
            filename: str = None,
            output_dir: str = ''):

    """

    :param model:
    :param x: input data to predict
    :param save_format: format of the output file that predicted values are save. Possible values:
        - None: dont save predicted values
        - "json": save as json file
        - "npy": save as numpy file
    :param filename:
    :param output_dir: output directory for the output file

    :return: predicted values as np.array
    """

    y_pred = model.predict(x)

    if save_format:
        if save_format not in ['npy', 'json']:
            raise ValueError('invalid save_format {}'.format(save_format))

        if (filename is None) or (filename == ""):
            filename = '{}_y_pred'.format(model.name)

        if not filename.endswith(save_format):
            filename = format('{}.{}'.format(filename, save_format))

        file_path = os.path.join(output_dir, filename)

        if save_format == "npy":
            np.save(file_path, y_pred)
        else:
            save_json(y_pred.flatten().tolist(), file_path)

    return y_pred.flatten()
