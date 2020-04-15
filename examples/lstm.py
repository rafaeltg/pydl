import os
import numpy as np
from pydl.model_selection import r2_score, rmse
from pydl.models.layers import LSTM, Dense, Dropout
from pydl.models import RNN, load_model, model_from_json
from dataset import create_multivariate_dataset


# Example of valid configurations of layers
models = [
    RNN(
        name='lstm1',
        layers=[
            LSTM(units=32, activation='relu')
        ],
        epochs=200
    ),
    RNN(
        name='lstm2',
        layers=[
            LSTM(units=32, activation='relu'),
            Dense(units=16)
        ],
        epochs=200
    ),
    RNN(
        name='lstm3',
        layers=[
            LSTM(units=32, activation='relu'),
            Dense(units=16),
            Dropout(0.1)
        ],
        epochs=200
    ),
    RNN(
        name='lstm4',
        layers=[
            LSTM(units=32, activation='relu'),
            LSTM(units=32, activation='relu')
        ],
        epochs=200
    ),
    RNN(
        name='lstm5',
        layers=[
            LSTM(units=32, activation='relu'),
            LSTM(units=32, activation='relu'),
            Dense(units=8)
        ],
        epochs=200
    )
]


def run_lstm():

    """
        LSTM example
    """

    n_features = 4
    n_steps = 1

    x, y, x_test, y_test = create_multivariate_dataset(
        size=n_steps * 100,
        n_steps=n_steps,
        n_features=n_features)

    for model in models:
        print('\nTraining')
        model.fit(x=x, y=y)

        print(model.summary())

        train_score = model.score(x=x, y=y)
        print('Train score = {}'.format(train_score))

        test_score = model.score(x=x_test, y=y_test)
        print('Test score = {}'.format(test_score))

        print('Predicting test data')
        y_test_pred = model.predict(x_test)

        y_test_rmse = rmse(y_test, y_test_pred)
        print('y_test RMSE = {}'.format(y_test_rmse))

        y_test_r2 = r2_score(y_test, y_test_pred)
        print('y_test R2 = {}'.format(y_test_r2))

        print('Saving model')
        model.save('models/')
        model.save_json('models/')
        model.save_weights('models/')

        assert os.path.exists('models/{}.h5'.format(model.name))
        assert os.path.exists('models/{}.json'.format(model.name))
        assert os.path.exists('models/{}_weights.h5'.format(model.name))

        print('Loading model from .h5 file')
        model1 = load_model('models/{}.h5'.format(model.name))
        assert isinstance(model1, RNN)
        assert model1.name == model.name

        print('Validating train score')
        np.testing.assert_equal(train_score, model1.score(x=x, y=y))

        print('Validating test score')
        np.testing.assert_equal(test_score, model1.score(x=x_test, y=y_test))

        print('Validating predicted test data')
        y_test_pred_new = model1.predict(x_test)
        np.testing.assert_allclose(y_test_pred, y_test_pred_new, atol=1e-6)
        np.testing.assert_equal(y_test_rmse, rmse(y_test, y_test_pred_new))
        np.testing.assert_equal(y_test_r2, r2_score(y_test, y_test_pred_new))

        del model1

        print('Loading model from json and weights files')
        model2 = model_from_json('models/{}.json'.format(model.name),
                                 weights_filepath='models/{}_weights.h5'.format(model.name),
                                 compile=True)
        assert isinstance(model2, RNN)
        assert model2.name == model.name

        print('Validating train score')
        np.testing.assert_equal(train_score, model2.score(x=x, y=y))

        print('Validating test score')
        np.testing.assert_equal(test_score, model2.score(x=x_test, y=y_test))

        print('Validating predicted test data')
        y_test_pred_new = model2.predict(x_test)
        np.testing.assert_allclose(y_test_pred, y_test_pred_new, atol=1e-6)
        np.testing.assert_equal(y_test_rmse, rmse(y_test, y_test_pred_new))
        np.testing.assert_equal(y_test_r2, r2_score(y_test, y_test_pred_new))

        del model2


if __name__ == '__main__':
    run_lstm()
