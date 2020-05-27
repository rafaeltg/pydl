import numpy as np
from ._base_transformer import TransformerMixin, SerializerMixin

__all__ = [
    'StandardScaler'
]


class StandardScaler(TransformerMixin, SerializerMixin):

    def __init__(self, mean=None, std=None, name=None):
        self.mean = mean or []
        self.std = std or []
        self.name = name or 'scaler'

    def fit(self, X, y, **kwargs):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X, y=None):
        X -= self.mean
        X /= self.std
        return X, y

    def get_config(self) -> dict:
        return {
            'name': self.name,
            'mean': list(self.mean),
            'std': list(self.std)
        }
