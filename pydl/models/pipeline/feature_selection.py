from ._base_transformer import TransformerMixin, SerializerMixin

__all__ = [
    'BaseFilterSelect'
]


class BaseFilterSelect(TransformerMixin, SerializerMixin):

    def __init__(self, indexes, name=None):
        self.indexes = indexes
        self.name = name or 'feature_selector'

    def get_support(self):
        return self.indexes

    def transform(self, X, y=None):
        return X[..., self.indexes], y

    def get_config(self) -> dict:
        return {
            'indexes': self.indexes,
            'name': self.name
        }
