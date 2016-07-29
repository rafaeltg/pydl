import numpy as np
import tensorflow as tf

from models.autoencoder_models.autoencoder import Autoencoder
from models.nnet_models.hidden_layer import HiddenLayer


class DenoisingAutoencoder(Autoencoder):

    """ Implementation of (Sparse) Denoising Autoencoder using TensorFlow.
    """

    def __init__(self,
                 model_name='dae',
                 main_dir='dae/',
                 n_hidden=256,
                 enc_act_func='tanh',
                 dec_act_func='none',
                 cost_func='rmse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 corr_type='masking',
                 corr_scale=0.1,
                 corr_keep_prob=0.9,
                 rho=0.001,
                 n_beta=3.0,
                 n_lambda=0.0001,
                 verbose=0,
                 seed=-1):

        """
        :param n_hidden: number of hidden units
        :param enc_act_func: Activation function for the encoder. ['tanh', 'sigmoid', 'relu']
        :param dec_act_func: Activation function for the decoder. ['tanh', 'sigmoid', 'relu', 'none']
        :param cost_func: Cost function. ['rmse', 'cross_entropy', 'softmax_cross_entropy', 'sparse']
        :param num_epochs: Number of epochs
        :param batch_size: Size of each mini-batch
        :param opt: Which TensorFlow optimizer to use. ['gradient_descent', 'momentum', 'ada_grad', 'adam', 'rms_prop']
        :param learning_rate: Initial learning rate
        :param momentum: Momentum parameter
        :param corr_type: Type of input corruption. ["masking", "gaussian"]
        :param corr_scale:
        :param corr_keep_prob:
        :param rho:
        :param n_beta:
        :param n_lambda:
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        print('{} __init__'.format(__class__.__name__))

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         n_hidden=n_hidden,
                         enc_act_func=enc_act_func,
                         dec_act_func=dec_act_func,
                         cost_func=cost_func,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         rho=rho,
                         n_beta=n_beta,
                         n_lambda=n_lambda,
                         verbose=verbose,
                         seed=seed)

        # Validations
        assert corr_type in ["masking", "gaussian"]
        assert corr_scale > 0 if corr_type == 'gaussian' else True
        assert 0 <= corr_keep_prob <= 1.0 if corr_type == 'masking' else True

        self._corr_input = None
        self.corr_type = corr_type
        self.corr_scale = corr_scale
        self.corr_keep_prob = corr_keep_prob

        print('Done {} __init__'.format(__class__.__name__))

    def _create_encoding_layer(self):

        """ Create the encoding layer based on the corrupted version of X.
        :return: self
        """

        print('Creating {} encoding layer'.format(self.model_name))

        self._corrupt_input()

        self._encode_layer = HiddenLayer(input_layer=self._corr_input,
                                         hidden_units=self.n_hidden,
                                         act_func=self.enc_act_func,
                                         name_scope='encode_layer')

        self._encode = self._encode_layer.get_output()

        print('Done creating {} encoding layer'.format(self.model_name))

    def _corrupt_input(self):

        """ Create the activation function corrupting X based on corruption type
        :return: self
        """

        print('Corrupting Input Data')

        with tf.name_scope('corrupted_input'):
            if self.corr_type == 'masking':
                self._corr_input = tf.nn.dropout(self._input, self.corr_keep_prob)

            elif self.corr_type == 'gaussian':
                self._corr_input = tf.add(self._input,
                                          tf.constant(self.corr_scale)*tf.random_normal((np.int_(self._input.get_shape()[1]),)))