from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import utils.utilities as utils
from models.base.unsupervised_model import UnsupervisedModel
from models.nnet_models.hidden_layer import HiddenLayer


class Autoencoder(UnsupervisedModel):

    """ Implementation of Autoencoders using TensorFlow.
    """

    def __init__(self,
                 model_name='ae',
                 main_dir='ae/',
                 n_hidden=256,
                 enc_act_func='tanh',
                 dec_act_func='none',
                 cost_func='rmse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 rho=0.01,
                 n_beta=3.0,
                 n_lambda=0.0001,
                 verbose=0,
                 seed=-1):

        """
        :param n_hidden: number of hidden units
        :param enc_act_func: Activation function for the encoder. ['tanh', 'sigmoid', 'relu']
        :param dec_act_func: Activation function for the decoder. ['tanh', 'sigmoid', 'relu', 'none']
        :param cost_func: Cost function. ['rmse', 'cross_entropy', 'softmax_cross_entropy', 'sparse']
        :param num_epochs: Number of epochs for training
        :param batch_size: Size of each mini-batch
        :param opt: Which tensorflow optimizer to use. ['gradient_descent', 'momentum', 'ada_grad', 'adam', 'rms_prop']
        :param learning_rate: Initial learning rate
        :param momentum: Momentum parameter
        :param rho:
        :param n_beta:
        :param n_lambda:
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         cost_func=cost_func,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        self.logger.info('{} __init__'.format(__class__.__name__))

        # Validations
        assert n_hidden > 0
        assert enc_act_func in utils.valid_act_functions
        assert dec_act_func in utils.valid_act_functions
        assert cost_func in utils.valid_unsupervised_cost_functions
        assert rho > 0 if cost_func == 'sparse' else True
        assert n_beta > 0 if cost_func == 'sparse' else True
        assert n_lambda > 0 if cost_func == 'sparse' else True

        self.n_hidden = n_hidden
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func

        # Sparse Variables
        if cost_func == 'sparse':
            self.sparse_rho = tf.constant(rho, name='sparse-rho')
            self.sparse_beta = tf.constant(n_beta, name='sparse-beta')
            self.sparse_lambda = tf.constant(n_lambda, name='sparse-lambda')

        self._encode_layer = None
        self._decode_layer = None

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, n_inputs):

        """ Create the encoding and the decoding layers of the autoencoder.
        :return: self
        """

        self._create_encoding_layer()
        self._create_decoding_layer(n_inputs)

    def _create_encoding_layer(self):

        """ Create the encoding layer of the autoencoder.
        :return: self
        """

        self.logger.info('Creating {} encoding layer'.format(self.model_name))

        self._encode_layer = HiddenLayer(input_layer=self._input,
                                         hidden_units=self.n_hidden,
                                         act_func=self.enc_act_func,
                                         name_scope='encode_layer')

        self._encode = self._encode_layer.get_output()

        self.logger.info('Done creating {} encoding layer'.format(self.model_name))

    def _create_decoding_layer(self, n_inputs):

        """ Create the decoding layers of the autoencoder.
        :param n_inputs:
        :return: self
        """

        self.logger.info('Creating {} decoding layer'.format(self.model_name))

        self._decode_layer = HiddenLayer(input_layer=self._encode,
                                         hidden_units=n_inputs,
                                         act_func=self.dec_act_func,
                                         name_scope='decode_layer')

        self._model_output = self._decode_layer.get_output()

        self.logger.info('Done creating {} decoding layer'.format(self.model_name))

    def _create_cost_node(self, ref_input):

        """
        :param ref_input:
        :return: self
        """

        if self.cost_func == 'sparse':
            # Based on Andrew Ng's description as in
            # https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf

            def KL_Div(rho, rho_hat):
                invrho = tf.sub(utils.ONE, rho)
                invrhohat = tf.sub(utils.ONE, rho_hat)
                logrho = tf.add(tf.mul(rho, tf.log(tf.div(rho, rho_hat))),
                                tf.mul(invrho, tf.log(tf.div(invrho, invrhohat))))
                return logrho

            cost_j = tf.reduce_mean(tf.nn.l2_loss(tf.sub(self._model_output, ref_input)))

            cost_reg = tf.mul(self.sparse_lambda,
                              tf.add(tf.nn.l2_loss(self._encode_layer.get_weitgths()),
                                     tf.nn.l2_loss(self._decode_layer.get_weitgths())))

            rho_hat = tf.reduce_mean(self._encode, 0)
            cost_sparse = tf.mul(self.sparse_beta, tf.reduce_sum(KL_Div(self.sparse_rho, rho_hat)))

            with tf.name_scope("cost"):
                self.cost = tf.add(tf.add(cost_j, cost_reg), cost_sparse)
                _ = tf.scalar_summary("sparse", self.cost)

        else:

            super()._create_cost_node(ref_input)

    def _train_model(self, train_set, valid_set):

        """Train the model.
        :param train_set: training set
        :param valid_set: validation set
        :return: self
        """

        self.logger.info('Training {}'.format(self.model_name))

        for i in range(self.num_epochs):
            self.logger.info('Training epoch {}...'.format(i))

            np.random.shuffle(train_set)
            batches = [_ for _ in utils.gen_batches(train_set, self.batch_size)]

            for batch in batches:
                self.tf_session.run(self.optimizer, feed_dict={self._input: batch})

            if valid_set is not None:
                self._run_validation_cost_and_summaries(i, {self._input: valid_set})

        self.logger.info('Done Training {}'.format(self.model_name))

    def get_model_parameters(self, graph=None):

        """ Return the model parameters in the form of numpy arrays.
        :param graph: tensorflow graph object
        :return: model parameters
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                params = {
                    'enc_w': self._encode_layer.get_weights().eval(),
                    'enc_b': self._encode_layer.get_biases().eval()
                    # don't care about decoder params
                }

        return params
