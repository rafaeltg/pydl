import keras.layers as kl
from ..base import SupervisedModel
from ..layers import LSTM, CuDNNLSTM, Dense, Conv1D, MaxPooling1D, Flatten, TimeDistributed, Dropout


class CNNLSTM(SupervisedModel):

    """
        A CNN network that learns input features and an LSTM that interprets them.
    """

    @classmethod
    def is_valid_layer(cls, layer):
        return isinstance(layer, Conv1D) or \
               isinstance(layer, MaxPooling1D) or \
               isinstance(layer, Flatten) or \
               isinstance(layer, LSTM) or \
               isinstance(layer, CuDNNLSTM) or \
               isinstance(layer, Dense) or \
               isinstance(layer, Dropout) or \
               cls.is_valid_time_distributed_layer(layer)

    @classmethod
    def is_valid_time_distributed_layer(cls, td_layer):
        return isinstance(td_layer, TimeDistributed) and \
               (isinstance(td_layer.layer, Conv1D) or
                isinstance(td_layer.layer, MaxPooling1D) or
                isinstance(td_layer.layer, Flatten))

    @classmethod
    def check_layers_config(cls, layers: list):
        layers_config = []

        for i, l in enumerate(layers):
            if not cls.is_valid_layer(l):
                raise TypeError('{} is not a valid layer for CNNLSTM model.'.format(l.__class__))

            if cls.is_valid_time_distributed_layer(l):
                layers_config.append(l)
            elif isinstance(l, Conv1D) or isinstance(l, MaxPooling1D) or isinstance(l, Flatten):
                layers_config.append(TimeDistributed(l))
            else:
                if isinstance(l, kl.RNN) and isinstance(layers_config[-1], TimeDistributed):
                    layers_config.append(TimeDistributed(Flatten()))

                layers_config.append(l)

        return layers_config
