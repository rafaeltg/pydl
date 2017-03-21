import os
from .utils import get_model, get_input_data, load_data
from pydl.models import SupervisedModel
from pydl.model_selection import available_metrics
from pydl.models.utils.utilities import load_model, save_json


def score(config, output):
    """
    """

    m = load_model(get_model(config))

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')

    metrics = config['metrics'] if 'metrics' in config else []

    results = {}
    if isinstance(m, SupervisedModel):
        y = load_data(data_set, 'data_y')

        results[m.get_loss_func()] = m.score(x, y)
        if len(metrics) > 0:
            y_pred = m.predict(x)
            for metric in metrics:
                results[metric] = available_metrics[metric](y, y_pred)

    else:
        results[m.get_loss_func()] = m.score(x)
        if len(metrics) > 0:
            x_rec = m.reconstruct(m.transform(x))
            for metric in metrics:
                results[metric] = available_metrics[metric](x, x_rec)

    # Save results into a JSON file
    save_json(results, os.path.join(output, m.name+'_scores.json'))