import os
import numpy as np
from ..models import model_from_config, save_json
from ..hyperopt import Node, CMAES
from ..model_selection import CV
from .fit import fit


def obj_fn(s, x, y):
    try:
        model = model_from_config(config=s)

        result = CV(method='split', test_size=0.2).run(
            model=model,
            x=x,
            y=y,
            scoring='rmse')['rmse']['mean']
    except:
        result = np.nan

    return result


def optimization(search_space: Node,
                 x, y,
                 x0: list = None,
                 cmaes_params: dict = None,
                 output_dir: str = '',
                 refit_best_model: bool = True,
                 best_model_fit_kwargs: dict = None,
                 save_to_json: bool = False,
                 features: list = None,
                 max_threads: int = 1):

    """
    Hyperparameters optimization and/or feature selection


    :param search_space: search space variables definition
    :param x: input data for optimization process
    :param y: target values for optimization process
    :param x0: optional list of initial guess of minimum solution. Values from 0 to 1.
        Default value is [1] * search_space.size
    :param cmaes_params: optional parameters for CMAES algorithm
    :param output_dir: output directory for the output files
    :param refit_best_model:
    :param best_model_fit_kwargs:
    :param save_to_json:
    :param features:
    :param max_threads:

    :return: best model configuration
    """

    if search_space.size == 0:
        raise ValueError('empty search space')

    if x0 is not None:
        if len(x0) != search_space.size:
            raise ValueError('invalid x0. It must contains {} elements'.format(search_space.size))

    opt = CMAES(
        verb_filenameprefix=os.path.join(output_dir, 'cmaes/out_'),
        **(cmaes_params or {})
    )

    result = opt.fmin(
        search_space=search_space,
        x0=x0,
        obj_func=obj_fn,
        args=(x, y),
        max_threads=max_threads
    )

    s = search_space.get_value(result[0])

    model = model_from_config(config=s)
    model.name += '_best'

    if refit_best_model:
        fit(model=model, x=x, y=y, **(best_model_fit_kwargs or {}))

    if features and hasattr(model, 'get_support'):
        sup = getattr(model, 'get_support')()
        if sup:
            features = list(np.array(features)[sup])

    res = {
        'model': {
            'class_name': model.__class__.__name__,
            'config': model.get_config()
        },
        'best_fit_func': result[1],
        'best_x': list(result[0]),
        'best_features': features
    }

    if save_to_json:
        save_json(res, os.path.join(output_dir, '{}_opt.json'.format(model.name)))

    return res, model
