import json
from keras.utils.io_utils import H5Dict
from ..json import save_json


class LinearMixin:

    def get_config(self):
        raise NotImplemented

    def to_json(self, **kwargs) -> str:
        m = {
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }
        return json.dumps(m, **kwargs)

    def save_json(self, filepath: str = None):
        save_json(self, filepath)

    def save(self, filepath: str = None):
        with H5Dict(filepath, mode='w') as h5dict:
            model_config = {
                'class_name': self.__class__.__name__,
                'config': self.get_config()
            }
            h5dict['model_config'] = json.dumps(model_config).encode('utf-8')

    @property
    def built(self):
        raise NotImplemented
