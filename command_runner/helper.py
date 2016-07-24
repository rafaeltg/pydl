import numpy as np
import utils.utilities as utils

from models.base.supervised_model import SupervisedModel
from models.base.unsupervised_model import UnsupervisedModel
from utils import datasets


######################
#  Flags definition  #
######################

def default_flags(flags):

    """
    :param flags:
    :return: self
    """

    flags.DEFINE_string('train_dataset', '', 'Path to train set file (.npy or .csv).')
    flags.DEFINE_string('test_dataset', '', 'Path to test set file (.npy or .csv).')
    flags.DEFINE_string('valid_dataset', '', 'Path to validation set file (.npy or .csv).')
    flags.DEFINE_boolean('restore_model', False, 'If true, restore previous model corresponding to model name.')


def set_global_flags_unsupervised(flags):

    """
    :param flags:
    :return: self
    """

    default_flags(flags)
    flags.DEFINE_boolean('save_encode', False, 'Whether to encode and save the training set.')


def set_global_flags_supervised(flags):

    """
    :param flags:
    :return: self
    """

    default_flags(flags)
    flags.DEFINE_string('train_labels', '', 'Path to train labels file (.npy or .csv).')
    flags.DEFINE_string('test_labels', '', 'Path to test labels file (.npy or .csv).')
    flags.DEFINE_string('valid_labels', '', 'Path to validation labels file (.npy or .csv).')


def model_flags(model_name, flags):

    """
    :param flags:
    :return: self
    """

    flags.DEFINE_string('model_name', model_name, 'Model name.')
    flags.DEFINE_string('main_dir', model_name+'/', 'Directory to store data relative to the algorithm.')

    flags.DEFINE_integer('num_epochs', 20, 'Number of training epochs.')
    flags.DEFINE_integer('batch_size', 500, 'Size of each training mini-batch.')
    flags.DEFINE_float('xavier_init', 1, 'Value for the constant in xavier weights initialization.')
    flags.DEFINE_string('opt', 'adam', 'Optmization algorithm. {}'.format(utils.valid_optimization_functions))
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')

    flags.DEFINE_integer('verbose', 1, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
    flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')


def set_supervised_model_flags(model_name, flags):

    """
    """

    set_global_flags_supervised(flags)
    model_flags(model_name, flags)

    flags.DEFINE_string('enc_act_func', 'relu', 'Activation function for the hidden layers. {}'.format(utils.valid_act_functions))
    flags.DEFINE_string('dec_act_func', 'none', 'Activation function for the output layer. {}'.format(utils.valid_act_functions))
    flags.DEFINE_string('cost_func', 'rmse', 'Cost function to be minimized. {}'.format(utils.valid_supervised_cost_functions))
    flags.DEFINE_float('dropout', 1.0, 'Hidden layers dropout.')
    flags.DEFINE_string('task', 'regression', 'Which type of task to perform. ["regression", "classification"]')


def set_unsupervised_model_flags(model_name, flags):

    """
    """

    set_global_flags_unsupervised(flags)
    model_flags(model_name, flags)

    flags.DEFINE_string('enc_act_func', 'relu', 'Activation function for the encoder layer. {}'.format(utils.valid_act_functions))
    flags.DEFINE_string('dec_act_func', 'none', 'Activation function for the decode layer. {}'.format(utils.valid_act_functions))
    flags.DEFINE_string('cost_func', 'rmse', 'Cost function. {}'.format(utils.valid_unsupervised_cost_functions))



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
    testX  = data.test.data
    testY  = data.test.target
    validX = data.validation.data
    validY = data.validation.target

    # Train the model
    model.fit(trainX, trainY, validX, validY, restore_previous_model=global_params['restore_model'])

    if testX is not None:
        test_cost = model.calc_total_cost(testX, testY)
        print('Test Cost = {}'.format(test_cost))
        #metrics


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
    testX  = data.test.data
    validX = data.validation.data

    # Train the model
    model.fit(trainX, validX, restore_previous_model=global_params['restore_model'])

    if global_params['save_encode']:
        # Encode the training data and store it
        enc_train_x = model.transform(trainX)
        enc_test_x = model.transform(validX)
        enc_valid_x = model.transform(testX)

    if testX is not None:
        test_cost = model.calc_total_cost(testX)
        print('Test Cost = {}'.format(test_cost))
