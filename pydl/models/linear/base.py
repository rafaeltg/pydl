import json
from keras.utils.io_utils import H5Dict


class LinearMixin:

    def get_config(self):
        raise NotImplemented

    def to_json(self, **kwargs) -> str:
        m = {
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }
        return json.dumps(m, **kwargs).encode('utf-8')

    def save_json(self, filepath: str = None):
        cfg_json = self.to_json(indent=2, ensure_ascii=False, separators=(',', ': '))
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cfg_json)

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
