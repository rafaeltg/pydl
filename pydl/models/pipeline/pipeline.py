import os
import h5py
import json
import numpy as np
from ..utils import check_filepath, model_from_config
from ..json import save_json


class Pipeline:

    def __init__(self,
                 steps: list = [],
                 name: str = 'pipeline'):
        self.steps = steps
        self._validate_steps()
        self.name = name or steps[-1].name
        self.named_steps = {s.name.lower(): s for s in steps if s != 'passthrough'}
        self._support = None

    def _validate_steps(self):
        if len(self.steps) == 0:
            raise ValueError('missing steps')

        if not all([(isinstance(s, str) and (s == 'passthrough')) or hasattr(s, 'fit_transform') for s in
                    self.steps[:-1]]):
            raise ValueError('invalid steps')

        if not (hasattr(self.steps[-1], 'fit') and
                hasattr(self.steps[-1], 'predict') and
                hasattr(self.steps[-1], 'score')):
            raise ValueError('invalid estimator')

    def get_support(self):
        if not self.built:
            return None

        for k, step in self.named_steps.items():
            if hasattr(step, 'get_support'):
                self._support = getattr(step, 'get_support')()

        return self._support

    @property
    def built(self):
        return self.steps[-1].built

    def fit(self, x, y):
        self._support = [True] * x.shape[1]

        for s in self.steps[:-1]:
            if s != 'passthrough':
                x, y = s.fit_transform(x, y)

        return self.steps[-1].fit(x, y)

    def predict(self, x):
        x, _ = self.transform(x)
        return self.steps[-1].predict(x)

    def score(self, x, y):
        x, y = self.transform(x, y)
        loss = self.steps[-1].score(x=x, y=y)
        return loss[0] if isinstance(loss, list) else loss

    def transform(self, x, y=None):
        for s in self.steps[:-1]:
            if s != 'passthrough':
                x, y = s.transform(x, y)

        return x, y

    def set_params(self, **params):
        for k, v in params.items():
            keys = k.split('__')
            if keys[0] in self.named_steps:
                self.named_steps[keys[0]].set_params(**{keys[1]: v})

    def get_config(self) -> dict:
        config = {
            'name': self.name,
            'steps': [
                {
                    'class_name': s.__class__.__name__,
                    'config': s.get_config()
                } for s in self.named_steps.values()
            ]
        }

        return config

    @classmethod
    def from_config(cls, config: dict, custom_objects: dict = None):
        cfg = config.copy()

        if 'steps' in config:
            cfg['steps'] = [
                model_from_config(config=s, custom_objects=custom_objects)
                for s in config['steps'] if not isinstance(s, str)
            ]

        return cls(**cfg)

    def to_json(self, **kwargs) -> str:
        m = {
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }
        return json.dumps(m, **kwargs)

    def save_json(self, filepath: str = None):
        filepath = check_filepath(filepath, self.name, 'json')
        save_json(self, filepath)

    def save(self, filepath=None):
        filepath = check_filepath(filepath, self.name, 'h5')

        with h5py.File(filepath, mode='w') as hf:
            model_config = hf.create_group('model_config')
            model_config['class_name'] = self.__class__.__name__
            config = model_config.create_group('config')

            config['name'] = self.name

            if len(self.steps) > 1:
                config['steps'] = np.asarray([s.to_json().encode('utf-8') for s in self.steps[:-1]])

            if hasattr(self.steps[-1], 'save'):
                try:
                    with h5py.File('tmpmodel.h5', mode='w') as model_hf:
                        self.steps[-1].save(filepath=model_hf)

                    with open('tmpmodel.h5', mode='rb') as in_file:
                        config['estimator'] = str(in_file.read())

                except Exception as e:
                    raise e

                finally:
                    os.remove('tmpmodel.h5')

    def compile(self, optimizer, loss=None):
        if hasattr(self.steps[-1], 'compile'):
            self.steps[-1].compile(
                optimizer=optimizer,
                loss=loss
            )

    def get_optimizer(self):
        if hasattr(self.steps[-1], 'get_optimizer'):
            return self.steps[-1].get_optimizer()
        return None

    def get_loss_func(self):
        if hasattr(self.steps[-1], 'get_loss_func'):
            return self.steps[-1].get_loss_func()
        return None

    def __len__(self):
        """
        Returns the length of the Pipeline
        """
        return len(self.steps)
