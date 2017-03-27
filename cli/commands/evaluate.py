import os
from .utils import get_input_data, load_data, get_model
from pydl.model_selection import get_scorer
from pydl.models.utils import save_json, load_model
from pydl.datasets.time_series import acf


def evaluate(config, output):
    """
    """

    m = load_model(get_model(config))

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')
    y = load_data(data_set, 'data_y') if 'data_y' in data_set else None

    assert 'scoring' in config, 'Missing scoring'
    scoring = config['scoring']

    if y is None:
        results = dict([(s, get_scorer(s)(m, x)) for s in scoring])
    else:
        results = dict([(s, get_scorer(s)(m, x, y)) for s in scoring])

    if 'errors_acf' in config:
        errs = y - m.predict(x)
        acf_params = config['errors_acf']
        acfs, conf_lvl = acf(errs, **acf_params)
        results['errors_acf'] = {
            'acfs': acfs,
            'conf_lvl': conf_lvl
        }

    save_json(results, os.path.join(output, m.name+'_eval.json'))