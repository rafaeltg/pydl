import json
from keras.utils.io_utils import H5Dict
from ..utils import check_filepath, model_from_config
from .reshape import Reshaper


class Pipeline:

    def __init__(self,
                 estimator,
                 name: str = 'pipeline',
                 features: list = None,
                 reshaper: Reshaper = None):
        self._estimator = estimator
        self.name = name or estimator.name
        self._features = features
        self._reshaper = reshaper
        self._validate_params()

    def _validate_params(self):
        if self._estimator is None:
            raise ValueError('missing estimator')

        self._validate_features()

        if self._reshaper is not None:
            assert isinstance(self._reshaper, Reshaper), ValueError("'reshaper' must be a instance of Reshaper")

    def _validate_features(self):
        if self._features is not None:
            if not isinstance(self._features, list):
                raise ValueError("'features' must be a list")

            if len(self._features) == 0:
                raise ValueError("'features' cannot be empty")

            if not all([isinstance(f, int) for f in self._features]):
                raise ValueError("'features' must be a list of integers")

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value
        self._validate_features()

    @property
    def built(self):
        return self._estimator.built

    def fit(self, x, y):
        x, y = self.transform(x, y)
        self._estimator.fit(x, y)

    def predict(self, x):
        x, _ = self.transform(x)
        return self._estimator.predict(x)

    def score(self, x, y):
        x, y = self.transform(x, y)
        loss = self._estimator.evaluate(x=x, y=y)
        return loss[0] if isinstance(loss, list) else loss

    def transform(self, x, y=None):
        if self._features is not None:
            x = x[..., self._features]

        if self._reshaper is not None:
            x, y = self._reshaper.reshape(x, y)

        return x, y

    def get_config(self) -> dict:
        config = {
            'name': self.name
        }

        if self._features is not None:
            config['features'] = self._features

        if self._reshaper is not None:
            config['reshaper'] = {
                'class_name': self._reshaper.__class__.__name__,
                'config': self._reshaper.get_config()
            }

        config['estimator'] = {
            'class_name': self._estimator.__class__.__name__,
            'config': self._estimator.get_config()
        }

        return config

    @classmethod
    def from_config(cls, config: dict, custom_objects: dict = None):
        if 'reshaper' in config:
            config['reshaper'] = model_from_config(config=config.get('reshaper', {}), custom_objects=custom_objects)
        config['estimator'] = model_from_config(config=config.get('estimator', {}), custom_objects=custom_objects)
        return cls(**config)

    def to_json(self, **kwargs) -> str:
        m = {
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }
        return json.dumps(m, **kwargs)

    def save_json(self, filepath: str = None):
        filepath = check_filepath(filepath, self.name, 'json')
        cfg_json = self.to_json(sort_keys=True, indent=2, ensure_ascii=False, separators=(',', ': '))
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cfg_json)

    def save(self, filepath=None):
        filepath = check_filepath(filepath, self.name, 'h5')

        with H5Dict(filepath, mode='w') as h5dict:
            model_config = h5dict['model_config']
            model_config['class_name'] = self.__class__.__name__
            config = model_config['config']

            config['name'] = self.name

            if self._features is not None:
                config['features'] = json.dumps(self._features).encode('utf-8')

            if self._reshaper is not None:
                config['reshaper'] = self._reshaper.to_json()

            if hasattr(self._estimator, 'save'):
                estimator = config['estimator']
                self._estimator.save(estimator.data)

    def compile(self, optimizer, loss=None):
        if hasattr(self._estimator, 'compile'):
            self._estimator.compile(
                optimizer=optimizer,
                loss=loss)

    def get_optimizer(self):
        if hasattr(self._estimator, 'get_optimizer'):
            return self._estimator.get_optimizer()
        return None

    def get_loss_func(self):
        if hasattr(self._estimator, 'get_loss_func'):
            return self._estimator.get_loss_func()
        return None
