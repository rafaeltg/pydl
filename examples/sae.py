import os
import numpy as np

from pydl.models.autoencoder_models.stacked_autoencoder import StackedAutoencoder
from validator.cv_metrics import mape


def run_dae():

    """
        Stacked Autoencoder example
    """

    # Create fake dataset
    input_dim = 20
    train_size = 2000
    test_size = 200
    x_train = np.random.normal(loc=0, scale=1, size=(train_size, input_dim))
    y_train = np.random.normal(loc=0, scale=1, size=(train_size, 1))
    x_test = np.random.normal(loc=0, scale=1, size=(test_size, input_dim))
    y_test = np.random.normal(loc=0, scale=1, size=(test_size, 1))

    print('Creating Stacked Autoencoder')
    sae = StackedAutoencoder(
        layers=[32, 16],
        num_epochs=[200],
    )

    print('Training')
    sae.fit(x_train=x_train, y_train=y_train)

    train_score = sae.score(x=x_train, y=y_train)
    print('Train score = {}'.format(train_score))

    test_score = sae.score(x=x_test, y=y_test)
    print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = sae.predict(data=x_test)
    print('Predicted y_test shape = {}'.format(y_test_pred.shape))
    assert y_test_pred.shape == (test_size, 1)

    y_test_mape = mape(y_test, y_test_pred)
    print('MAPE for y_test forecasting = {}'.format(y_test_mape))

    print('Saving model')
    sae.save_model('/home/rafael/models/sae.h5')
    assert os.path.exists('/home/rafael/models/sae.h5')

    print('Loading model')
    sae_new = StackedAutoencoder(
        layers=[32, 16],
        num_epochs=[200],
    )

    sae_new.load_model('/home/rafael/models/sae.h5')

    print('Calculating train score')
    assert train_score == sae_new.score(x=x_train, y=y_train)

    print('Calculating test score')
    assert test_score == sae_new.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = sae_new.predict(data=x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating MAPE')
    assert y_test_mape == mape(y_test, y_test_pred_new)


if __name__ == '__main__':
    run_dae()
