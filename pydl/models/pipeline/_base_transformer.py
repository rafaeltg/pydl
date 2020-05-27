import json


class SerializerMixin:
    def get_config(self) -> dict:
        pass

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def to_json(self, **kwargs) -> str:
        m = {
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }
        return json.dumps(m, **kwargs)


class TransformerMixin:

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X, y=None):
        raise NotImplemented

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X=X, y=y, **fit_params).transform(X=X, y=y)

    def set_params(self, **params):
        for k, v in params.items():
            if k in self.__dict__:
                setattr(self, k, v)
