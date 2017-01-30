from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN

from pydl.models.base.supervised_model import SupervisedModel


class RNN(SupervisedModel):

    """ Generic Recurrent Neural Network
    """

    def __init__(self,
                 name='rnn',
                 layers=list([50, 50]),
                 cell_type='lstm',
                 stateful=True,
                 time_steps=1,
                 enc_act_func='tanh',
                 dec_act_func='linear',
                 loss_func='mse',
                 num_epochs=100,
                 batch_size=1,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 dropout=0.1,
                 verbose=0,
                 seed=42):

        """
        :param name: Name of the model.
        :param layers: Number of hidden units in each layer.
        :param cell_type: Recurrent layers type. ["lstm", "gru", "simple"]
        :param stateful: Whether the recurrent network is stateful or not.It means that the states
            computed for the samples in one batch will be reused as initial states for the samples
            in the next batch.
        :param time_steps:
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

        self.cell_type = cell_type
        self.stateful = stateful
        self.time_steps = time_steps

        super().__init__(name=name,
                         layers=layers,
                         enc_act_func=enc_act_func,
                         dec_act_func=dec_act_func,
                         loss_func=loss_func,
                         dropout=dropout,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def validate_params(self):
        super().validate_params()
        assert self.cell_type in ["lstm", "gru", "simple"], 'Invalid cell type!'
        assert self.time_steps > 0, "time_steps must be grater than zero!"

    def _create_layers(self, input_shape, n_output):

        """ Create the network layers
        :param n_output:
        :return: self
        """

        if self.cell_type == 'lstm':
            cell = LSTM
        elif self.cell_type == 'gru':
            cell = GRU
        else:
            cell = SimpleRNN

        b_size = self.batch_size if self.stateful else None

        # Hidden layers
        for i, l in enumerate(self.layers):

            self._model.add(cell(output_dim=l,
                                 batch_input_shape=(b_size, input_shape[1], input_shape[2]),
                                 activation=self.enc_act_func,
                                 stateful=self.stateful,
                                 return_sequences=True if i < (len(self.layers)-1) else False))

            if self.dropout > 0:
                self._model.add(Dropout(p=self.dropout))

        # Output layer
        self._model.add(Dense(output_dim=n_output,
                              activation=self.dec_act_func))

    def _train_step(self, x_train, y_train, x_valid=None, y_valid=None):

        if self.stateful:
            for i in range(self.num_epochs):
                print('>> Epoch', i, '/', self.num_epochs)

                self._model.fit(x=x_train,
                                y=y_train,
                                batch_size=self.batch_size,
                                verbose=self.verbose,
                                nb_epoch=1,
                                shuffle=False,
                                validation_data=(x_valid, y_valid) if x_valid and y_valid else None)
                self._model.reset_states()
        else:
            super()._train_step(x_train, y_train, x_valid, y_valid)
