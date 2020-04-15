import os
import numpy as np
from pydl.model_selection import r2_score
from pydl.models.layers import LSTM
from pydl.models import RNN, load_model
from dataset import create_multivariate_dataset


def run_lstm_stateful():

    """
        Stateful LSTM example
    """

    n_features = 4
    n_steps = 1

    x, y, x_test, y_test = create_multivariate_dataset(
        size=n_steps * 100,
        n_steps=n_steps,
        n_features=n_features)

    print('Creating a stateful LSTM')
    model = RNN(
        layers=[
            LSTM(units=15, stateful=True, activation='relu')
        ],
        batch_size=1,
        epochs=200)

    print('Training')
    model.fit(x=x, y=y)

    train_score = model.score(x=x, y=y)
    print('Train score = {}'.format(train_score))

    test_score = model.score(x=x_test, y=y_test)
    print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = model.predict(x_test)
    print(y_test_pred)
    print('Predicted y_test shape = {}'.format(y_test_pred.shape))

    y_test_r2 = r2_score(y_test, y_test_pred)
    print('R2 for y_test forecasting = {}'.format(y_test_r2))

    print('Saving model')
    model.save('models/lstm_stateful.h5')
    model.save_json('models/lstm_stateful.json')
    assert os.path.exists('models/lstm_stateful.json')
    assert os.path.exists('models/lstm_stateful.h5')

    del model

    print('Loading model')
    model = load_model('models/lstm_stateful.h5')

    print('Calculating train score')
    assert train_score == model.score(x=x, y=y)

    print('Calculating test score')
    assert test_score == model.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = model.predict(x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating R2 for test set')
    assert y_test_r2 == r2_score(y_test, y_test_pred_new)


if __name__ == '__main__':
    run_lstm_stateful()
