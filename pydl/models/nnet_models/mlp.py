from keras.layers import Dense, Dropout
from keras.regularizers import l1_l2
from ..base import SupervisedModel


class MLP(SupervisedModel):

    """ Multi-Layer Perceptron
    """

    def __init__(self,
                 name='mlp',
                 layers=list(),
                 activation='relu',
                 out_activation='linear',
                 dropout=0,
                 l1_reg=0,
                 l2_reg=0,
                 **kwargs):

        """
        :param layers: List of layers in the network.
        """

        super().__init__(name=name,
                         layers=layers,
                         activation=activation,
                         out_activation=out_activation,
                         dropout=dropout,
                         l1_reg=l1_reg,
                         l2_reg=l2_reg,
                         **kwargs)

    def _create_layers(self, input_shape, n_output):

        """ Create the network layers
        :param input_shape:
        :param n_output:
        :return: self
        """

        # Hidden layers
        for i, l in enumerate(self.layers):
            self._model.add(Dense(units=l,
                                  input_shape=[input_shape[-1] if i == 0 else None],
                                  activation=self.activation[i],
                                  kernel_regularizer=l1_l2(self.l1_reg[i], self.l2_reg[i]),
                                  bias_regularizer=l1_l2(self.l1_reg[i], self.l2_reg[i])))

            if self.dropout[i] > 0:
                self._model.add(Dropout(rate=self.dropout[i]))

        # Output layer
        self._model.add(Dense(units=n_output, activation=self.out_activation))