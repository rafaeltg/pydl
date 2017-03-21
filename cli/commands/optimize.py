from .utils import get_input_data, load_data, get_obj_fn, get_optimizer
from pydl.hyperopt import HyperOptModel, hp_space_from_json


def optimize(config, output):
    """
    """

    # Get hp_space
    assert 'hp_space' in config, 'Missing hyperparameters space definition'
    space = hp_space_from_json(config['hp_space'])

    # Get data
    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')
    y = load_data(data_set, 'data_y') if 'data_y' in data_set else None

    # Get HyperOptModel
    assert 'opt' in config, 'Missing optimizer parameters!'
    opt = get_optimizer(config['opt'])
    obj_fn = get_obj_fn(config['opt'])

    opt_model = HyperOptModel(hp_space=space, fit_fn=obj_fn, opt=opt)
    result = opt_model.fit(x, y, retrain=True)

    print('\n>> Best params =', result['best_model_config'])
    print('\n>> Best fit =', result['opt_result'][1])

    # Save best model
    opt_model.best_model.save_model(output)
