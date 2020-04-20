import os
import json
import numpy as np
from pydl.ts import train_test_split
from pydl.model_selection import CV, r2_score, rmse
from pydl.models import Pipeline, load_model, hp_rnn, hp_mlp, hp_lstm, hp_dense, hp_pipeline, hp_reshaper3d, \
    model_from_config
from pydl.hyperopt import hp_choice, hp_int, hp_feature_selection, CMAES
from dataset import create_multivariate_data


def loss_fn(model_config, x, y):
    try:
        m = model_from_config(model_config)
        res = CV('split', test_size=0.2).run(m, x, y, scoring='rmse')['rmse']['mean']
    except:
        res = np.nan

    return res


def run_hp_pipeline():

    """
        Pipeline optimization example
    """

    n_features = 10

    data = create_multivariate_data(size=600, n_features=n_features)
    x, y, x_test, y_test = train_test_split(data[:, :-1], data[:, -1])

    print('Creating Pipeline')
    space = hp_choice([
        hp_pipeline(
            name='mlp',
            features=hp_feature_selection(n_features),
            estimator=hp_mlp(
                name='mlp',
                layers=hp_choice([
                    [
                        hp_dense(
                            units=hp_int(8, 32),
                            activation=hp_choice(['relu', 'tanh', 'sigmoid'])
                        )
                    ],
                    [
                        hp_dense(
                            units=hp_int(8, 32),
                            activation=hp_choice(['relu', 'tanh', 'sigmoid'])
                        ),
                        hp_dense(
                            units=hp_int(8, 32),
                            activation=hp_choice(['relu', 'tanh', 'sigmoid'])
                        )
                    ]
                ]),
                epochs=hp_int(10, 100)
            )
        ),
        hp_pipeline(
            name='lstm',
            features=hp_feature_selection(n_features),
            reshaper=hp_reshaper3d(
                n_steps=hp_int(1, 20)
            ),
            estimator=hp_rnn(
                name='lstm',
                layers=hp_choice([
                    [
                        hp_lstm(
                            units=hp_int(8, 32)
                        )
                    ],
                    [
                        hp_lstm(
                            units=hp_int(8, 32)
                        ),
                        hp_lstm(
                            units=hp_int(8, 32)
                        )
                    ]
                ]),
                epochs=hp_int(10, 100)
            )
        ),
    ])

    print('Setting up optimizer...')
    cmaes = CMAES(
        pop_size=10,
        max_iter=10,
        verbose=10)

    res = cmaes.fmin(
        search_space=space,
        obj_func=loss_fn,
        args=(x, y),
        max_threads=1)

    best_model_cfg = space.get_value(res[0])

    print(json.dumps(best_model_cfg, indent=4, separators=(',', ': ')))

    model = model_from_config(best_model_cfg)
    model_name = model.name

    print('Training')
    model.fit(x=x, y=y)

    train_score = model.score(x=x, y=y)
    print('Train {} = {}'.format(model.get_loss_func().upper(), train_score))

    test_score = model.score(x=x_test, y=y_test)
    print('Test {} = {}'.format(model.get_loss_func().upper(), test_score))

    print('Predicting test data')
    y_test_pred = model.predict(x_test)

    y_test_rmse = rmse(y_test, y_test_pred)
    print('y_test RMSE = {}'.format(y_test_rmse))

    y_test_r2 = r2_score(y_test, y_test_pred)
    print('y_test R2 = {}'.format(y_test_r2))

    print('Saving model')
    model.save('models/')
    model.save_json('models/')

    assert os.path.exists('models/{}.json'.format(model.name))
    assert os.path.exists('models/{}.h5'.format(model.name))

    del model

    print('Loading model from .h5 file')
    model = load_model('models/{}.h5'.format(model_name))
    assert isinstance(model, Pipeline)
    assert model.name == model_name

    print('Calculating train score')
    np.testing.assert_equal(train_score, model.score(x=x, y=y))

    print('Calculating test score')
    np.testing.assert_equal(test_score, model.score(x=x_test, y=y_test))

    print('Predicting test data')
    y_test_pred_new = model.predict(x_test)
    np.testing.assert_allclose(y_test_pred, y_test_pred_new, atol=1e-6)

    print('Calculating RMSE for test set')
    np.testing.assert_equal(y_test_rmse, rmse(y_test, y_test_pred_new))

    print('Calculating R2 for test set')
    np.testing.assert_equal(y_test_r2, r2_score(y_test, y_test_pred_new))


if __name__ == '__main__':
    run_hp_pipeline()
