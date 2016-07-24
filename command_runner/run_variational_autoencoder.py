import tensorflow as tf
import numpy as np

from models.autoencoder_models.variational_autoencoder import VariationalAutoencoder
import utils.utilities as utils
from utils import datasets

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', '', 'Which dataset to use.')
flags.DEFINE_string('model_name', 'vae', 'Model name.')
flags.DEFINE_string('main_dir', 'vae/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_boolean('save_encode', False, 'Whether to encode and save the training set.')

# Variational Autoencoder specific parameters
flags.DEFINE_string('n_hidden', '128,', 'Number of hidden units in each layer.')
flags.DEFINE_integer('n_z', 5, 'Latent space size')
flags.DEFINE_string('enc_act_func', 'sigmoid', 'Activation function for the encoder layer. {}'.format(utils.valid_act_functions))
flags.DEFINE_string('dec_act_func', 'none', 'Activation function for the decode layer. {}'.format(utils.valid_act_functions))
flags.DEFINE_string('cost_func', 'rmse', 'Cost function. {}'.format(utils.valid_unsupervised_cost_functions))
flags.DEFINE_integer('num_epochs', 30, 'Number of epochs for training.')
flags.DEFINE_integer('batch_size', 500, 'Size of each mini-batch.')
flags.DEFINE_integer('xavier_init', 1, 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('opt', 'adam', 'Optmizer algorithm. {}'.format(utils.valid_optimization_functions))
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')


# Validade input
assert FLAGS.dataset is not ''
assert FLAGS.model_name is not ''
assert FLAGS.main_dir is not ''

hidden_layers = utils.flag_to_list(FLAGS.n_hidden, 'int'),
assert len(hidden_layers) > 0
assert FLAGS.n_z > 0
assert FLAGS.enc_act_func in utils.valid_act_functions
assert FLAGS.dec_act_func in utils.valid_act_functions
assert FLAGS.cost_func in utils.valid_unsupervised_cost_functions
assert FLAGS.num_epochs > 0
assert FLAGS.batch_size > 0
assert FLAGS.opt in utils.valid_optimization_functions
assert FLAGS.learning_rate > 0

if FLAGS.opt == 'momentum':
    assert FLAGS.momentum > 0


def run_variational_autoencoder():

    """
    :param vae_params:
    :return: self
    """

    # Assertions

    # Read dataset
    trainX, trainY = datasets.load_csv(FLAGS.dataset, has_header=True)

    # Create the Autoencoder object
    vae = VariationalAutoencoder(model_name=FLAGS.model_name,
                                 main_dir=FLAGS.main_dir,
                                 n_hidden=hidden_layers,
                                 n_z=FLAGS.n_z,
                                 enc_act_func=FLAGS.enc_act_func,
                                 dec_act_func=FLAGS.dec_act_func,
                                 num_epochs=FLAGS.num_epochs,
                                 batch_size=FLAGS.batch_size,
                                 xavier_init=FLAGS.xavier_init,
                                 opt=FLAGS.opt,
                                 learning_rate=FLAGS.learning_rate,
                                 momentum=FLAGS.momentum,
                                 verbose=FLAGS.verbose,
                                 seed=FLAGS.seed)

    # Fit the model
    vae.fit(trainX, restore_previous_model=FLAGS.restore_previous_model)

    # Encode the training data and store it
    transformedX = vae.transform(trainX)

    cost = vae.calc_total_cost(trainX)
    print('Total Cost = {}'.format(cost))


if __name__ == '__main__':
    run_variational_autoencoder()

