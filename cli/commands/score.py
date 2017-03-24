import os
from .utils import get_input_data, load_data
from pydl.model_selection import get_scorer
from pydl.models.utils.utilities import save_json


def score(config, output):
    """
    """

    assert 'scoring' in config, 'Missing scoring!'
    scoring = config['scoring'] if 'scoring' in config else []

    data_set = get_input_data(config)
    actual_out = load_data(data_set, 'actual_out')
    pred_out = load_data(data_set, 'predict_out')

    results = dict([(s, get_scorer(s)(actual_out, pred_out)) for s in scoring])

    out_file = config['out_file'] if 'out_file' in config else 'scores.json'

    # Save results into a JSON file
    save_json(results, os.path.join(output, out_file))
