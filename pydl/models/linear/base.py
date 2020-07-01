import h5py
import json
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

    def save(self, filepath = None):
        if filepath is None:
            filepath = self.__class__.__name__.lower()

        def _save(f):
            model_config = f.create_group('model_config')
            model_config['class_name'] = self.__class__.__name__
            model_config['config'] = json.dumps(self.get_config(), ensure_ascii=False).encode('utf-8')

        if isinstance(filepath, str):
            with h5py.File(filepath, mode='w') as hf:
                _save(hf)
        else:
            _save(filepath)

    @property
    def built(self):
        raise NotImplemented
