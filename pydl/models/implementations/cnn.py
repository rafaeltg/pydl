from keras.layers import Flatten
from ..base import SupervisedModel
from ..layers import Dense, Conv1D, Conv2D, MaxPooling1D, SpatialDropout1D, Dropout


class CNN(SupervisedModel):

    """
        Generic Convolutional Neural Network
    """

    @classmethod
    def is_valid_layer(cls, layer):
        return cls.is_valid_conv_layer(layer) or \
               isinstance(layer, Flatten) or \
               isinstance(layer, Dropout) or \
               isinstance(layer, Dense)

    @classmethod
    def is_valid_conv_layer(cls, layer):
        return isinstance(layer, Conv1D) or \
               isinstance(layer, Conv2D) or \
               isinstance(layer, MaxPooling1D) or \
               isinstance(layer, SpatialDropout1D)

    @classmethod
    def check_layers_config(cls, layers: list):
        layers_config = []

        for i, l in enumerate(layers):
            if not cls.is_valid_layer(l):
                raise TypeError('{} is not a valid layer for CNN model.'.format(l.__class__))

            layers_config.append(l)

            if cls.is_valid_conv_layer(l) and \
                    ((i == len(layers) - 1) or
                     (not cls.is_valid_conv_layer(layers[i + 1]) and not isinstance(layers[i + 1], Flatten))):
                layers_config.append(Flatten())

        return layers_config
