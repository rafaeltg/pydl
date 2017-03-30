import os
from .utils import get_input_data, load_data, get_cv_config
from pydl.models import SupervisedModel
from pydl.model_selection import CV
from pydl.models.utils.utilities import load_model, save_json


def cv(config, output):
    """
    """

    m = load_model(config)

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')

    if isinstance(m, SupervisedModel):
        y = load_data(data_set, 'data_y')
    else:
        y = None

    # Get validation method
    method, params, scoring, max_threads = get_cv_config(config)
    results = CV(method=method, **params).run(model=m, x=x, y=y, scoring=scoring, max_threads=max_threads)

    # Save results into a JSON file
    save_json(results, os.path.join(output, m.name+'_cv.json'))
