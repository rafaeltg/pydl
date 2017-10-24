import os
import numpy as np
from pyts import mackey_glass, create_dataset
from pydl.model_selection.metrics import mape
from pydl.models import StackedAutoencoder, Autoencoder, load_model, save_model


def run_sae():

    """
        Stacked Autoencoder example
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

    print('Creating Stacked Autoencoder')
    sae = StackedAutoencoder(
        layers=[Autoencoder(n_hidden=32, enc_activation='relu'),
                Autoencoder(n_hidden=16, enc_activation='relu')],
        nb_epochs=100
    )

    print('Training')
    sae.fit(x_train=x_train, y_train=y_train)

    train_score = sae.score(x=x_train, y=y_train)
    print('Train score = {}'.format(train_score))

    test_score = sae.score(x=x_test, y=y_test)
    print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = sae.predict(x_test)
    print('Predicted y_test shape = {}'.format(y_test_pred.shape))
    assert y_test_pred.shape == y_test.shape

    y_test_mape = mape(y_test, y_test_pred)
    print('MAPE for y_test forecasting = {}'.format(y_test_mape))

    print('Saving model')
    save_model(sae, 'models/', 'sae')
    assert os.path.exists('models/sae.json')
    assert os.path.exists('models/sae.h5')

    print('Loading model')
    sae_new = load_model('models/sae.json')

    print('Calculating train score')
    assert train_score == sae_new.score(x=x_train, y=y_train)

    print('Calculating test score')
    assert test_score == sae_new.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = sae_new.predict(x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating MAPE')
    assert y_test_mape == mape(y_test, y_test_pred_new)


if __name__ == '__main__':
    run_sae()
