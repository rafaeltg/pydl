import numpy as np

from models.base.supervised_model import SupervisedModel
from models.base.unsupervised_model import UnsupervisedModel
from utils import datasets


#########################
#  Execution functions  #
#########################

def run_supervised_model(model, global_params):

    """
    :param model:
    :param global_params:
    :return: self
    """

    assert isinstance(model, SupervisedModel)
    assert global_params['train_dataset'] != ''
    assert global_params['train_labels'] != ''

    # Read dataset
    data = datasets.load_datasets(train_dataset=global_params['train_dataset'],
                                  train_labels=global_params['train_labels'],
                                  test_dataset=global_params['test_dataset'],
                                  test_labels=global_params['test_labels'],
                                  valid_dataset=global_params['valid_dataset'],
                                  valid_labels=global_params['valid_labels'],
                                  has_header=True)

    trainX = data.train.data
    trainY = data.train.target
    validX = data.validation.data
    validY = data.validation.target
    testX  = data.test.data
    testY  = data.test.target

    # Train the model
    model.fit(trainX, trainY, validX, validY, restore_previous_model=global_params['restore_model'])

    if testX is not None:
        test_cost = model.calc_total_cost(testX, testY)
        print('Test Cost = {}'.format(test_cost))

        # Save the predictions of the model
        if global_params['save_predictions']:
            np.save(global_params['save_predictions'], model.predict(testX))


def run_unsupervised_model(model, global_params):

    """
    :param model:
    :param global_params:
    :return: self
    """

    assert isinstance(model, UnsupervisedModel)
    assert global_params['train_dataset'] != ''

    # Read dataset
    data = datasets.load_datasets(train_dataset=global_params['train_dataset'],
                                  test_dataset=global_params['test_dataset'],
                                  valid_dataset=global_params['valid_dataset'],
                                  has_header=True)

    trainX = data.train.data
    validX = data.validation.data
    testX  = data.test.data

    # Train the model
    model.fit(trainX, validX, restore_previous_model=global_params['restore_model'])

    if global_params['save_encode_train']:
        np.save(global_params['save_encode_train'], model.transform(trainX))

    if (validX is not None) and global_params['save_encode_valid']:
        np.save(global_params['save_encode_valid'], model.transform(validX))

    if testX is not None:
        test_cost = model.calc_total_cost(testX)
        print('Test Cost = {}'.format(test_cost))

        if global_params['save_encode_test']:
            np.save(global_params['save_encode_test'], model.transform(testX))
