import keras.layers as kl
from keras.callbacks import Callback
from ..base import SupervisedModel
from ..layers import Dense, Flatten, Dropout, ConvLSTM2D, MaxPooling1D, BatchNormalization


class ResetStatesCallback(Callback):

    def __init__(self, max_len):
        self.counter = 0
        self.max_len = max_len
        super().__init__()

    def on_batch_begin(self, batch, logs={}):
        if self.counter == 0:
            self.model.reset_states()
            self.counter = self.max_len-1
        else:
            self.counter -= 1


class RNN(SupervisedModel):

    """
        Generic Recurrent Neural Network
    """

    @classmethod
    def is_valid_layer(cls, layer):
        return isinstance(layer, kl.RNN) or \
               isinstance(layer, Flatten) or \
               isinstance(layer, Dropout) or \
               isinstance(layer, Dense) or \
               isinstance(layer, MaxPooling1D) or \
               isinstance(layer, BatchNormalization)

    @classmethod
    def check_layers_config(cls, layers: list):
        layers_config = []

        for i, l in enumerate(layers):
            if not cls.is_valid_layer(l):
                raise TypeError('{} is not a valid layer for RNN model.'.format(l.__class__))

            if isinstance(l, ConvLSTM2D):
                if i == len(layers) - 1:
                    layers_config.append(l)
                    layers_config.append(Flatten())
                else:
                    l.return_sequences = any([isinstance(layer, ConvLSTM2D) for layer in layers[(i+1):]])
                    layers_config.append(l)

                    if isinstance(layers[i + 1], Dense) or isinstance(layers[i + 1], Dropout):
                        layers_config.append(Flatten())

            else:
                if isinstance(l, kl.RNN):
                    l.return_sequences = (i < (len(layers) - 1)) and \
                                     (not (isinstance(layers[i+1], Dense) or isinstance(layers[i+1], Dropout)))

                layers_config.append(l)

        return layers_config

    def _get_input_shape(self, input_shape):
        return (self.batch_size if self.stateful else None, ) + input_shape[1:]

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=None,
            verbose=None,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=False,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):

        if self.stateful:
            self._callbacks.append(ResetStatesCallback(x.shape[0]))

        super().fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            **kwargs)
