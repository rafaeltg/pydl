import json

from sklearn.preprocessing import MinMaxScaler

from pydl.model_selection import CV
from pydl.datasets import mackey_glass, create_dataset
from pydl.hyperopt import *


def run_optimizer():

    """
        CMAES Optimizer example
    """

    print('Creating dataset')
    # Create time series data
    ts = mackey_glass(sample_len=2000)
    # Normalize the dataset
    ts = MinMaxScaler(feature_range=(0, 1)).fit_transform(ts)
    x, y = create_dataset(ts, look_back=10)

    print('Creating MLP ConfigOptimizer')
    space = hp_space({
        'model': {
            'class_name': 'MLP',
            'config': {
                'layers': [hp_int(10, 1000), hp_int(10, 1000)],
                'dropout': [hp_float(0, 0.3), hp_float(0, 0.3)],
                'activation': hp_choice(['relu', 'tanh', 'sigmoid'])
            }
        }
    })

    print('Creating Fitness Function')
    fit_fn = CVObjectiveFunction()

    print('Creating CMAES optimizer')
    opt = CMAESOptimizer(pop_size=4, max_iter=2)

    print('Creating HyperOptModel...')
    model = HyperOptModel(hp_space=space, fit_fn=fit_fn, opt=opt)

    print('Optimizing!')
    res = model.fit(x, y, max_threads=2)

    print('Best parameters:')
    best_params = res['best_model_config']
    print(json.dumps(best_params, indent=4, separators=(',', ': ')))

    print('Test RMSE of the best model = {}'.format(res['opt_result'][1]))


if __name__ == '__main__':
    run_optimizer()
