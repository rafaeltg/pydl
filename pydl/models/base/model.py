import keras.models as km
import keras.optimizers as k_opt
from keras.callbacks import EarlyStopping
from ..utils import *
from ..json import save_json


class Model(km.Sequential):

    """
    Class representing an abstract Model.
    """

    def __init__(self,
                 name: str = None,
                 loss_func: str = 'mse',
                 epochs: int = 1,
                 batch_size: int = None,
                 opt: str = 'adam',
                 early_stopping: bool = False,
                 verbose: int = 0,
                 **kwargs):

        """
        :param name: Name of the model.
        :param loss_func:
        :param epochs:
        :param batch_size:
        :param opt:
        :param early_stopping: stop training when a monitored quantity has stopped improving
        :param verbose: Level of verbosity. 0 - silent, 1 - print.

        Optional parameters:
        :param learning_rate:
        :param momentum:
        :param patient: Early Stopping parameter. Number of epochs with no improvement after which training will
            be stopped.
        :param min_delta: Early Stopping parameter. Minimum change in the monitored quantity to qualify as
            an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
        """

        super().__init__(name=name if name else self.__class__.__name__)

        self.loss_func = loss_func
        self.epochs = epochs
        self.batch_size = batch_size

        self.opt = opt
        self.learning_rate = kwargs.get('learning_rate', None)
        self.momentum = kwargs.get('momentum', None)

        # Early stopping callback
        self.early_stopping = early_stopping
        if early_stopping:
            self.patient = int(kwargs.get('patient', 2))
            self.min_delta = float(kwargs.get('min_delta', 0.1))
            self._callbacks = [EarlyStopping(min_delta=self.min_delta, patience=self.patient)]
        else:
            self._callbacks = []

        self.verbose = verbose
        self.validate_params()

    def validate_params(self):
        assert isinstance(self.epochs, int) and self.epochs > 0, 'Invalid number of training epochs'

        if self.batch_size:
            assert isinstance(self.batch_size, int) and self.batch_size > 0, 'Invalid batch size'

        if isinstance(self.loss_func, str):
            assert self.loss_func in valid_loss_functions, 'Invalid loss function'

        assert self.opt in valid_optimizers, 'Invalid optimizer'

        if self.learning_rate:
            assert self.learning_rate > 0, 'Invalid learning rate'

        if self.momentum:
            assert self.momentum > 0, 'Invalid momentum rate'

        assert isinstance(self.early_stopping, bool), 'Invalid early_stopping value'
        if self.early_stopping:
            assert self.min_delta > 0, 'Invalid min_delta value'
            assert self.patient > 0, 'Invalid patient value'

    def set_params(self, **params):
        for k, v in params.items():
            if k in self.__dict__:
                setattr(self, k, v)

    def get_loss_func(self):
        return self.loss_func if isinstance(self.loss_func, str) else self.loss_func.__name__

    def get_optimizer(self):
        config = {}

        if self.learning_rate:
            config['lr'] = self.learning_rate

        if self.momentum:
            config['momentum'] = self.momentum

        return k_opt.get({
            'class_name': self.opt,
            'config': config
        })

    def save(self,
             filepath=None,
             overwrite: bool = True,
             include_optimizer: bool = True):

        if filepath is None or isinstance(filepath, str):
            filepath = check_filepath(filepath, self.name, 'h5')

        super().save(
            filepath=filepath,
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format='h5')

    def save_json(self, filepath: str = None):
        filepath = check_filepath(filepath, self.name, 'json')
        save_json(self, filepath)

    def save_weights(self, filepath, overwrite=True):
        super().save_weights(
            filepath=check_filepath(filepath, self.name + '_weights', 'h5'),
            overwrite=overwrite)

    def get_config(self):
        config = super().get_config()
        config['loss_func'] = self.loss_func
        config['epochs'] = self.epochs
        config['batch_size'] = self.batch_size
        config['opt'] = self.opt
        config['learning_rate'] = self.learning_rate
        config['momentum'] = self.momentum

        config['early_stopping'] = self.early_stopping
        if self.early_stopping:
            config['patient'] = self.patient
            config['min_delta'] = self.min_delta

        config['verbose'] = self.verbose
        return config

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        model = super().from_config(config, custom_objects)
        model.__dict__.update(config)
        return model

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

        self.compile(
            optimizer=self.get_optimizer(),
            loss=self.loss_func)

        # By default use 10% of training data for testing
        if self.early_stopping and ((validation_split == 0.) and (validation_data is None)):
            validation_split = 0.1

        if callbacks is not None:
            self._callbacks.extend(callbacks)

        super().fit(
            x=x,
            y=y,
            batch_size=batch_size if batch_size else self.batch_size,
            epochs=epochs if epochs else self.epochs,
            verbose=verbose if verbose else self.verbose,
            callbacks=self._callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            **kwargs)

    def score(self, x, y=None):
        """
        :param x: Input data
        :param y: Target values
        """

        loss = super().evaluate(
            x=x,
            y=y,
            batch_size=self.batch_size,
            verbose=self.verbose)

        return loss[0] if isinstance(loss, list) else loss
