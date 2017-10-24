import os
import numpy as np
from pyts import mackey_glass, create_dataset
from pydl.model_selection.metrics import r2_score
from pydl.models import RNN, load_model, save_model


def run_lstm_stateful():

    """
        Stateful LSTM example
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

    print('Creating a stateful LSTM')
    lstm = RNN(layers=[50, 50],
               stateful=True,
               time_steps=1,
               cell_type='lstm',
               dropout=[0.1, 0.1],
               nb_epochs=100,
               batch_size=1,
               early_stopping=True,
               min_delta=1e-5,
               patient=15)

    print('Training')
    lstm.fit(x_train=x_train, y_train=y_train, valid_split=.05)

    train_score = lstm.score(x=x_train, y=y_train)
    print('Train score = {}'.format(train_score))

    test_score = lstm.score(x=x_test, y=y_test)
    print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = lstm.predict(x_test)

    assert y_test_pred.shape == y_test.shape

    y_test_r2 = r2_score(y_test, y_test_pred)
    print('R² for y_test forecasting = {}'.format(y_test_r2))

    print('Saving model')
    save_model(lstm, 'models/', 'lstm_stateful')
    assert os.path.exists('models/lstm_stateful.json')
    assert os.path.exists('models/lstm_stateful.h5')

    print('Loading model')
    lstm_new = load_model('models/lstm_stateful.json')

    print('Calculating train score')
    assert train_score == lstm_new.score(x=x_train, y=y_train)

    print('Calculating test score')
    assert test_score == lstm_new.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = lstm_new.predict(x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating R²')
    assert y_test_r2 == r2_score(y_test, y_test_pred_new)


if __name__ == '__main__':
    run_lstm_stateful()
