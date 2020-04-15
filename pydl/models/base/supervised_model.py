import numpy as np
from copy import deepcopy
from .model import Model
from ..utils import *
from ..layers import Dense


class SupervisedModel(Model):

    """
        Class representing an abstract Supervised Model
    """

    def __init__(self,
                 layers: list = None,
                 out_activation: str = 'linear',
                 **kwargs):

        """
        :param layers: hidden layers.
        :param out_activation: output layer activation function.
        """

        self.out_activation = out_activation
        super().__init__(**kwargs)

        if layers:
            for layer in self.check_layers_config(layers):
                self.add(layer)

    def validate_params(self):
        super().validate_params()

        if self.out_activation:
            assert self.out_activation in valid_act_functions, 'Invalid output activation function'

    def build_model(self, input_shape, n_output=1):
        """ Creates the computational graph for the Supervised Model.
        :param input_shape: shape of the input data
        :param n_output: number of output values.
        """

        # Add output layer
        self.add(Dense(units=n_output, activation=self.out_activation, name='dense_out'))

        self.build(self._get_input_shape(input_shape))

    def _get_input_shape(self, input_shape):
        return (None, ) + input_shape[1:]

    @classmethod
    def is_valid_layer(cls, layer):
        raise NotImplementedError

    @classmethod
    def check_layers_config(cls, layers: list):
        assert all([cls.is_valid_layer(l) for l in layers])
        return layers

    @classmethod
    def from_config(cls, config: dict, custom_objects: dict = None):
        layers = [model_from_config(l, custom_objects) for l in config['layers']]
        layers = cls.check_layers_config(layers)

        c_objs = get_custom_objects()
        if custom_objects and isinstance(custom_objects, dict):
            c_objs.update(custom_objects)

        cfg = deepcopy(config)
        cfg['layers'] = [
            {
                'class_name': l.__class__.__name__,
                'config': l.get_config()
            } for l in layers
        ]
        return super().from_config(cfg, c_objs)

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

        if len(np.shape(y)) == 1:
            y = np.reshape(y, (len(y), 1))

        if not self.built:
            self.build_model(x.shape, y.shape[-1])

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
