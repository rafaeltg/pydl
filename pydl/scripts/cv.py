import os
from ..model_selection import CV
from ..models.json import save_json


def cv(model,
       x, y,
       cv_params: dict = None,
       scoring: list = None,
       save_to_json: bool = True,
       output_dir: str = ''):

    """
    Cross-validation analysis

    :param model:
    :param x: input data
    :param y: target values
    :param cv_params:
    :param scoring: list of metrics used to evaluate the model in each test fold
    :param save_to_json:
    :param output_dir: output directory for the json file

    :return: cv result
    """

    result = CV(**cv_params).run(
        model=model,
        x=x,
        y=y,
        scoring=scoring)

    if save_to_json:
        save_json(result, os.path.join(output_dir, '{}_cv.json'.format(model.name)))

    return result
