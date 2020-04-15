import os
import numpy as np
from pydl.model_selection import r2_score, rmse
from pydl.models.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout
from pydl.models import CNNLSTM, load_model, model_from_json
from pydl.ts.transform import split_sequences
from dataset import create_multivariate_data, train_test_split


# Example of valid configurations of layers
models = [
    CNNLSTM(
        name='cnnlstm1',
        layers=[
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(units=16)
        ],
        epochs=100
    ),
    CNNLSTM(
        name='cnnlstm2',
        layers=[
            Conv1D(filters=32, kernel_size=3),
            MaxPooling1D(pool_size=2),
            LSTM(units=5)
        ],
        epochs=100
    ),
    CNNLSTM(
        name='cnnlstm3',
        layers=[
            Conv1D(filters=32, kernel_size=3),
            Conv1D(filters=32, kernel_size=3),
            LSTM(units=16)
        ],
        epochs=100
    ),
    CNNLSTM(
        name='cnnlstm4',
        layers=[
            Conv1D(filters=32, kernel_size=3),
            Conv1D(filters=32, kernel_size=3),
            MaxPooling1D(pool_size=2),
            LSTM(units=5)
        ],
        epochs=100
    ),
    CNNLSTM(
        name='cnnlstm5',
        layers=[
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(units=16),
            Dense(units=10)
        ],
        epochs=100
    ),
    CNNLSTM(
        name='cnnlstm6',
        layers=[
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(units=16),
            Dense(units=10),
            Dropout(0.1)
        ],
        epochs=100
    )
]


def run_cnnlstm():

    """
        CNNLSTM example
    """

    n_features = 4
    n_steps = 10
    n_seq = 3

    data = create_multivariate_data(size=n_seq * n_steps * 100, n_features=n_features)
    x, y = split_sequences(data[:, :-1], data[:, -1], n_steps)
    x, y = split_sequences(x, y, n_seq)
    x, y, x_test, y_test = train_test_split(x, y)

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
        assert isinstance(model1, CNNLSTM)
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
        assert isinstance(model2, CNNLSTM)
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
    run_cnnlstm()
