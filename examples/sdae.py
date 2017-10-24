import os
import numpy as np
import matplotlib.pyplot as plt
from pyts import lorenz, create_dataset
from pydl.model_selection.metrics import r2_score
from pydl.models import StackedAutoencoder, DenoisingAutoencoder, load_model, save_model


def run_sdae():

    """
        Stacked Denoising Autoencoder example
    """

    # Create time series data
    ts = lorenz(n=2000)

    # reshape into X=t and Y=t+1
    look_back = 10
    x, y = create_dataset(ts, look_back)

    # split into train and test sets
    train_size = int(len(x) * 0.8)
    x_train, y_train = x[0:train_size], y[0:train_size]
    x_test, y_test = x[train_size:len(x)], y[train_size:len(y)]

    print('Creating Stacked Denoising Autoencoder')
    sdae = StackedAutoencoder(
        layers=[DenoisingAutoencoder(n_hidden=32,
                                     enc_activation='relu',
                                     corr_param=0.1),
                DenoisingAutoencoder(n_hidden=16,
                                     enc_activation='relu',
                                     corr_param=0.1)
                ],
        nb_epochs=300
    )

    print('Training')
    sdae.fit(x_train=x_train, y_train=y_train)

    train_score = sdae.score(x=x_train, y=y_train)
    print('Train score = {}'.format(train_score))

    test_score = sdae.score(x=x_test, y=y_test)
    print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = sdae.predict(x_test)
    print('Predicted y_test shape = {}'.format(y_test_pred.shape))
    assert y_test_pred.shape == y_test.shape

    y_test_mape = r2_score(y_test, y_test_pred)
    print('R2 for y_test forecasting = {}'.format(y_test_mape))

    print('Saving model')
    save_model(sdae, 'models/', 'sdae')
    assert os.path.exists('models/sdae.json')
    assert os.path.exists('models/sdae.h5')

    print('Loading model')
    sdae_new = load_model('models/sdae.json')

    print('Calculating train score')
    assert train_score == sdae_new.score(x=x_train, y=y_train)

    print('Calculating test score')
    assert test_score == sdae_new.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = sdae_new.predict(x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating R2')
    assert y_test_mape == r2_score(y_test, y_test_pred_new)

    # Actual x Predicted
    plt.figure(1)
    plt.plot(y_test, color='red', label='Actual')
    plt.plot(y_test_pred, color='blue', label='Predicted')
    plt.show(block=True)


if __name__ == '__main__':
    run_sdae()
