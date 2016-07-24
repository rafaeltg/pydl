import tensorflow as tf
import numpy as np
import utils.utilities as utils
import models.Model as model

from models.autoencoder_models.autoencoder import Autoencoder
from models.nnet_models import nn_layer


class VariationalAutoencoder(Autoencoder):

    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and realized by multi-layer perceptrons. The VAE can be learned end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    __NAME = 'VariationalAutoencoder'

    def __init__(self,
                 model_name='vae',
                 main_dir='vae/',
                 n_hidden=list([300, 300]),
                 n_z=30,
                 enc_act_func='relu',
                 dec_act_func='none',
                 num_epochs=10,
                 batch_size=10,
                 xavier_init=1,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 verbose=0,
                 seed=-1):

        """
        :param n_hidden: number of hidden units
        :param n_z: latent space size
        :param enc_act_func: Activation function for the encoder. ['tanh', 'sigmoid', 'relu']
        :param dec_act_func: Activation function for the decoder. ['tanh', 'sigmoid', 'none']
        :param num_epochs: Number of epochs
        :param batch_size: Size of each mini-batch
        :param xavier_init: Value of the constant for xavier weights initialization
        :param opt: Which tensorflow optimizer to use. ['gradient_descent', 'momentum', 'ada_grad', 'adam', 'rms_prop']
        :param learning_rate: Initial learning rate
        :param momentum: Momentum parameter
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        print('{} __init__'.format(self.__NAME))

        Autoencoder.__init__(self,
            model_name=model_name,
            main_dir=main_dir,
            n_hidden=n_hidden,
            enc_act_func=enc_act_func,
            dec_act_func=dec_act_func,
            num_epochs=num_epochs,
            batch_size=batch_size,
            xavier_init=xavier_init,
            opt=opt,
            learning_rate=learning_rate,
            momentum=momentum,
            verbose=verbose,
            seed=seed)

        self.n_z = n_z
        self.z_log_sigma_sq = None
        self.z = None

        print('Done {} __init__'.format(self.__NAME))


    def _create_layers(self, n_input):

        """ Create the encoding and the decoding layers of the network.
        :return: self
        """

        self._create_encoding_layers()
        self._create_decoding_layers(n_input)


    def _create_encoding_layers(self):

        """ Generate probabilistic encoder (recognition network),
            which maps inputs onto a normal distribution in latent space.
            The transformation is parametrized and can be learned.
        :return: self
        """

        print('Creating {} Encoding layers'.format(self.__NAME))

        self._encode_layer = []
        next_layer = self._input

        for l, layer in enumerate(self.n_hidden):
            node = nn_layer.NNetLayer(input_layer=next_layer,
                                      hidden_units=layer,
                                      act_function=self.enc_act_func,
                                      xavier_init=self.xavier_init,
                                      name_scope='enc_layer_{}'.format(l))

            next_layer = node.get_output()

            self._encode_layer.append(node)

        z_mean_layer = nn_layer.NNetLayer(input_layer=next_layer,
                                          hidden_units=self.n_z,
                                          act_function=None,
                                          xavier_init=self.xavier_init,
                                          name_scope='enc_out_mean')

        self._encode_layer.append(z_mean_layer)

        # Z mean
        self._encode = z_mean_layer.get_output()

        self.z_log_sigma_sq = nn_layer.NNetLayer(input_layer=next_layer,
                                                 hidden_units=self.n_z,
                                                 act_function=None,
                                                 xavier_init=self.xavier_init,
                                                 name_scope='enc_out_log_sigma_sq').get_output()

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal(tf.shape(self.z_log_sigma_sq), 0, 1, 'float')

        # z = mu + sigma*epsilon
        self.z = tf.add(self._encode,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        print('Done creating {} Encoding layers'.format(self.__NAME))


    def _create_decoding_layers(self, n_features):

        """ Generate probabilistic decoder (decoder network),
            which maps points in latent space onto a Bernoulli distribution in data space.
            The transformation is parametrized and can be learned.
        :return: self
        """

        print('Creating {} Decoding layers'.format(self.__NAME))

        self._decode_layer = []
        next_layer = self.z

        for l, layer in enumerate(self.n_hidden[::-1]):
            node = nn_layer.NNetLayer(input_layer=next_layer,
                                      hidden_units=layer,
                                      act_function=self.dec_act_func,
                                      xavier_init=self.xavier_init,
                                      name_scope='dec_layer_{}'.format(l))

            next_layer = node.get_output()

            self._decode_layer.append(node)

        # Reconstruction mean
        self._model_output = nn_layer.NNetLayer(input_layer=next_layer,
                                                hidden_units=n_features,
                                                name_scope='dec_out_mean').get_output()

        print('Done creating {} Decoding layers'.format(self.__NAME))


    def _create_cost_node(self, model_output, ref_input):

        """
        :return: self
        """

        """
        The loss is composed of two terms:
         1.) The reconstruction loss (the negative log probability of the input under the reconstructed
             Bernoulli distribution induced by the decoder in the data space).
             This can be interpreted as the number of "nats" required
             for reconstructing the input when the activation in latent is given.
        Adding 1e-10 to avoid evaluatio of log(0.0)
        """

        # cost
        #reconstr_loss = 0.5 * tf.reduce_sum(tf.square(tf.sub(self.x_reconstr_mean, self._input)))

        reconstr_loss = -tf.reduce_sum(tf.add(tf.mul(ref_input, tf.log(1e-10 + model_output)),
                                              tf.mul(tf.sub(utils.ONE, ref_input),
                                                     tf.log(tf.sub(tf.constant(1e-10 + 1),
                                                                   model_output)))), 1)

        """
        2) The latent loss, which is defined as the Kullback Leibler divergence
            between the distribution in latent space induced by the encoder on
            the data and some prior. This acts as a kind of regularizer.
            This can be interpreted as the number of "nats" required
            for transmitting the the latent space distribution given
            the prior.
        """

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                             - tf.square(self.z_mean)
                                             - tf.exp(self.z_log_sigma_sq), 1)

        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(reconstr_loss - latent_loss)   # average over batch
            _ = tf.scalar_summary("variational", cost)


    def get_model_parameters(self, graph=None):

        """ Return the model parameters in the form of numpy arrays.
        :param graph: tf graph object
        :return: model parameters
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)

                params = []

                for _, layer in enumerate(self._encode_layer):
                    params.append({
                        'w': layer.get_weights().eval(),
                        'b': layer.get_biases().eval()
                    })

        return params
