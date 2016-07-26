import numpy as np
import tensorflow as tf

import utils.utilities as utils


class NNetLayer(object):

    """ Class representing an abstract Neural Network Layer.
    """

    __NAME = 'NN_Layer'

    def __init__(self, 
                 input_layer,
                 hidden_units=256,
                 act_function='sigmoid',
                 dropout=1.0,
                 xavier_init=1,
                 name_scope='nn_layer',
                 init=True):

        """
        :param input_layer:
        :param hidden_units:
        :param act_function: Activation function. ['tanh', 'sigmoid', 'relu', 'none']
        :param dropout: The probability that each element is kept. Default = 1 (keep all)
        :param xavier_init:
        :param name_scope:
        :param init:
        """

        print('{} __init__ (hidden_units = {})'.format(self.__NAME, hidden_units))

        self._input = input_layer

        if init:

            self._w = tf.Variable(utils.xavier_init(np.int_(self._input.get_shape()[1]),
                                                    hidden_units, xavier_init), name='w-'+name_scope)

            self._b = tf.Variable(tf.truncated_normal([hidden_units], stddev=0.01), name='b-'+name_scope)

        else:

            self._w = tf.Variable(tf.zeros([np.int_(self._input.get_shape()[1]), hidden_units], 'float'))
            self._b = tf.Variable(tf.zeros([hidden_units], 'float'))

        with tf.name_scope(name_scope):
            
            self._output = utils.activate(act_function, tf.add(tf.matmul(self._input, self._w), self._b))

            if dropout != 1.0:
                self._output = tf.nn.dropout(self._output, tf.constant(dropout))

        print('Done {} __init__'.format(self.__NAME))

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
