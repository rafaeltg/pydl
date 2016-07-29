import numpy as np
import tensorflow as tf

import utils.utilities as utils


class HiddenLayer(object):

    """ Class representing an abstract Neural Network Layer.
    """

    def __init__(self,
                 input_layer,
                 hidden_units=256,
                 act_func='sigmoid',
                 dropout=1.0,
                 name_scope='nn_layer',
                 init=True):

        """
        :param input_layer: A tensor of shape (number_examples, n_input)
        :param hidden_units: Number of hidden units
        :param act_func: Activation function. ['tanh', 'sigmoid', 'relu', 'none']
        :param dropout: The probability that each element is kept. Default = 1 (keep all)
        :param name_scope:
        :param init:
        """

        print('{} __init__ (hidden_units = {})'.format(__class__.__name__, hidden_units))

        assert hidden_units > 0
        assert act_func in utils.valid_act_functions
        assert 0 <= dropout <= 1.0

        self._input = input_layer
        n_in = np.int_(self._input.get_shape()[1])

        if init:
            # Weights are initialized with values uniformly sampled from sqrt(-6./(n_in+n_hidden))
            # and sqrt(6./(n_in+n_hidden)). Optimal initialization of weights is dependent on the
            # activation function used (among other things). For example, results presented in [Xavier10]
            # suggest that you should use 4 times larger initial weights for sigmoid compared to tanh.
            # We have no info for other function, so we use the same as tanh.
            constant = 1 if act_func != 'sigmoid' else 4
            self._w = tf.Variable(tf.random_uniform((n_in, hidden_units),
                                                    minval=-constant * np.sqrt(6.0 / (n_in + hidden_units)),
                                                    maxval=constant * np.sqrt(6.0 / (n_in + hidden_units)),
                                                    dtype=tf.float32),
                                  name='w-'+name_scope)

            self._b = tf.Variable(tf.truncated_normal([hidden_units], stddev=0.01), name='b-'+name_scope)

        else:
            self._w = tf.Variable(tf.zeros([n_in, hidden_units], 'float'))
            self._b = tf.Variable(tf.zeros([hidden_units], 'float'))

        with tf.name_scope(name_scope):
            self._output = utils.activate(act_func, tf.add(tf.matmul(self._input, self._w), self._b))

            if dropout != 1.0:
                self._output = tf.nn.dropout(self._output, tf.constant(dropout))

        print('Done {} __init__'.format(__class__.__name__))

    def get_output(self):

        """
        :return: layer output
        """

        return self._output

    def get_weights(self):

        """
        :return: hidden weights
        """

        return self._w

    def set_weights(self, w):

        """
        :param w: weights values
        :return: self
        """

        self._w.assign(w)

    def get_biases(self):

        """
        :return: hidden biases
        """

        return self._b

    def set_biases(self, b):

        """
        :param b: biases values
        :return: self
        """

        self._b.assign(b)
