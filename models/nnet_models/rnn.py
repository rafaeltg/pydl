from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN

import utils.utilities as utils
from models.base.supervised_model import SupervisedModel


class RNN(SupervisedModel):

    """ Generic Recurrent Neural Network
    """

    def __init__(self,
                 model_name='rnn',
                 main_dir='rnn/',
                 cell_type='lstm',
                 layers=list([50, 50]),
                 enc_act_func='tanh',
                 dec_act_func='linear',
                 loss_func='mse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 dropout=0.2,
                 verbose=0,
                 seed=42):

        """
        :param model_name: Name of the model.
        :param main_dir: Directory to save the model data.
        :param cell_type: Recurrent layers type. ["lstm", "gru", "simple"]
        :param layers: Number of hidden units in each layer.
        :param enc_act_func: Activation function for the hidden layers.
        :param dec_act_func: Activation function for the output layer.
        :param loss_func: Cost function.
        :param num_epochs: Number of training epochs.
        :param batch_size: Size of each training mini-batch.
        :param opt: Optimizer function.
        :param learning_rate: Initial learning rate.
        :param momentum: Initial momentum value.
        :param dropout: The probability that each element is kept at each layer. Default = 1.0 (keep all).
        :param verbose: Level of verbosity. 0 - silent, 1 - print.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         loss_func=loss_func,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        self.logger.info('{} __init__'.format(__class__.__name__))

        # Validations
        assert cell_type in ["lstm", "gru", "simple"], 'Invalid cell type!'
        assert len(layers) > 0
        assert all([l > 0 for l in layers])
        assert enc_act_func in utils.valid_act_functions
        assert dec_act_func in utils.valid_act_functions
        assert 0 <= dropout <= 1.0

        if cell_type == 'lstm':
            self.cell = LSTM
        elif cell_type == 'gru':
            self.cell = GRU
        else:
            self.cell = SimpleRNN

        self.layers = layers
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.dropout = dropout

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, n_input, n_output):

        """ Create the network layers
        :param n_input:
        :param n_output:
        :return: self
        """

        # Hidden layers
        for n, l in enumerate(self.layers):

            if self.dropout < 1:
                self._model_layers = Dropout(p=self.dropout)(self._model_layers)

            self._model_layers = self.cell(output_dim=l,
                                           activation=self.enc_act_func,
                                           return_sequences=True if n < (len(self.layers)-1) else False)(self._model_layers)

        # Output layer
        if self.dropout < 1:
            self._model_layers = Dropout(p=self.dropout)(self._model_layers)

        self._model_layers = Dense(output_dim=n_output,
                                   activation=self.dec_act_func)(self._model_layers)

    def fit(self, x_train, y_train, x_valid, y_valid):

        if len(x_train.shape) != 3:
            x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

        if len(x_valid.shape) != 3:
            x_valid = np.reshape(x_valid, (x_valid.shape[0], 1, x_valid.shape[1]))

        super().fit(x_train, y_train, x_valid, y_valid)

    def predict(self, data):

        if len(data.shape) != 3:
            data = np.reshape(data, (data.shape[0], 1, data.shape[1]))

        super().predict(data)

    def score(self, x, y):

        if len(x.shape) != 3:
            x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

        super().score(x, y)
