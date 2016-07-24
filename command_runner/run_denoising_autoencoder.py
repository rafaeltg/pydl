import tensorflow as tf
import command_runner.cmd_flags
from command_runner.cmd_model_run import run_unsupervised_model
from models.autoencoder_models.denoising_autoencoder import DenoisingAutoencoder


# #################### #
#   Flags definition   #
# #################### #

flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
set_unsupervised_model_flags('dae', flags)

# Denoising Autoencoder specific parameters
flags.DEFINE_integer('n_hidden', 128, 'Number of hidden units of the Denoising Autoencoder.')
flags.DEFINE_string('corr_type', 'masking', 'Type of input corruption. ["masking", "gaussian"]')
flags.DEFINE_float('corr_scale', 0.1, '')
flags.DEFINE_float('corr_keep_prob', 0.8, '')
flags.DEFINE_float('rho', 0.01, '')
flags.DEFINE_float('n_beta', 3.0, '')
flags.DEFINE_float('n_lambda', 0.0001, '')


# Global parameters
global_params = {
    'train_dataset':     FLAGS.train_dataset,
    'test_dataset':      FLAGS.test_dataset,
    'valid_dataset':     FLAGS.valid_dataset,
    'restore_model':     FLAGS.restore_model,
    'save_encode_train': FLAGS.save_encode_train,
    'save_encode_valid': FLAGS.save_encode_valid,
    'save_encode_test':  FLAGS.save_encode_test,
}

# DAE parameters
dae_params = {
    'model_name':    FLAGS.model_name,
    'main_dir':      FLAGS.main_dir,
    'n_hidden':      FLAGS.n_hidden,
    'enc_act_func':  FLAGS.enc_act_func,
    'dec_act_func':  FLAGS.dec_act_func,
    'cost_func':     FLAGS.cost_func,
    'num_epochs':    FLAGS.num_epochs,
    'batch_size':    FLAGS.batch_size,
    'xavier_init':   FLAGS.xavier_init,
    'opt':           FLAGS.opt,
    'learning_rate': FLAGS.learning_rate,
    'momentum':      FLAGS.momentum,
    'corr_type':     FLAGS.corr_type,
    'corr_scale':    FLAGS.corr_scale,
    'corr_keep_prob':float(FLAGS.corr_keep_prob),
    'rho':           FLAGS.rho,
    'n_beta':        FLAGS.n_beta,
    'n_lambda':      FLAGS.n_lambda,
    'verbose':       FLAGS.verbose,
    'seed':          FLAGS.seed,
}


if __name__ == '__main__':

    # Create the Denoising Autoencoder object
    dae = DenoisingAutoencoder(**dae_params)

    run_unsupervised_model(dae, global_params)
