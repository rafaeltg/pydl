import os
import json
import numpy as np
from pyts import mackey_glass, create_dataset
from pydl.model_selection import r2_score
from pydl.models import RNN, RFM, load_model, save_model
from pydl.hyperopt import *


def run_rfm():

    """
        RFM example
    """

    # Create time series data
    ts = mackey_glass(n=2000)

    # reshape into X=t and Y=t+1
    look_back = 10
    x, y = create_dataset(ts, look_back)

    # split into train and test sets
    train_size = int(len(x) * 0.8)
    x_train, y_train = x[0:train_size], y[0:train_size]
    x_test, y_test = x[train_size:len(x)], y[train_size:len(y)]

    # reshape input to be [n_samples, time_steps, n_features]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    print('Creating a stateless LSTM')
    lstm = RNN(layers=[50, 50],
               dropout=[0., 0.2],
               stateful=False,
               time_steps=1,
               cell_type='lstm',
               nb_epochs=100,
               batch_size=50)

    rfm = RFM(fm=lstm, rm=lstm.copy())

    print('Training')
    rfm.fit(x_train=x_train, y_train=y_train)

    #train_score = lstm.score(x=x_train, y=y_train)
    #print('Train score = {}'.format(train_score))

    #test_score = lstm.score(x=x_test, y=y_test)
    #print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = rfm.predict(x_test)
    assert y_test_pred.shape == y_test.shape

    y_test_r2 = r2_score(y_test, y_test_pred)
    print('r2_score for y_test forecasting = {}'.format(y_test_r2))

    print('Saving model')
    save_model(rfm, 'models/', 'rfm')
    assert os.path.exists('models/rfm.json')
    assert os.path.exists('models/rfm_fm.h5')
    assert os.path.exists('models/rfm_rm.h5')

    print('Loading model')
    fem_new = load_model('models/rfm.json')

    #print('Calculating train score')
    #assert train_score == lstm_new.score(x=x_train, y=y_train)

    #print('Calculating test score')
    #assert test_score == lstm_new.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = fem_new.predict(x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating r2 score')
    assert y_test_r2 == r2_score(y_test, y_test_pred_new)


def optimize_rfm():

    """
        Optimizing RFM
    """

    print('Creating dataset')
    # Create time series data
    ts = mackey_glass(n=2000)
    x, y = create_dataset(ts, look_back=10)

    # reshape input to be [n_samples, time_steps, n_features]
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

    print('Creating RFM ConfigOptimizer')
    space = hp_space({
        'model': {
            'class_name': 'RFM',
            'config': {
                'name': 'rfm',
                'fm': {
                    'model': {
                        'class_name': 'RNN',
                        'config': {
                            'name': 'lstm_fm',
                            'cell_type': 'lstm',
                            'layers': [hp_int(10, 100)],
                            'dropout': [hp_float(0, 0.5)],
                            'stateful': False,
                        },
                    }
                },
                'rm': {
                    'model': {
                        'class_name': 'RNN',
                        'config': {
                            'name': 'lstm_fm',
                            'cell_type': 'lstm',
                            'layers': [hp_int(10, 100)],
                            'dropout': [hp_float(0, 0.5)],
                            'stateful': False,
                        },
                    }
                }
            }
        }
    })

    print('Creating Fitness Function')
    fit_fn = CVObjectiveFunction(scoring='r2')

    print('Creating HyperOptModel...')
    m = HyperOptModel(hp_space=space, fit_fn=fit_fn, opt='cmaes', opt_args={'pop_size': 8, 'max_iter': 1})

    print('Optimizing!')
    res = m.fit(x, y, max_threads=4)

    print('Best parameters:')
    best_params = res['best_model_config']
    print(json.dumps(best_params, indent=4, separators=(',', ': ')))

    print('Test R2 of the best model = {}'.format(res['opt_result'][1]))


if __name__ == '__main__':
    run_rfm()
    #optimize_rfm()
