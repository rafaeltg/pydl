import sklearn.decomposition as sd
from sklearn.utils.validation import check_is_fitted
from ._base_transformer import SerializerMixin


__all__ = [
    'PCA'
]


class PCA(sd.PCA, SerializerMixin):

    def __init__(self, n_components=None, whiten=False, name='pca'):
        super().__init__(
            n_components=n_components,
            whiten=whiten
        )
        self.name = name

    def get_config(self) -> dict:
        return super().get_params(deep=True)

    def fit_transform(self, X, y=None):
        self.n_components = min(self.n_components, X.shape[1] - 1)
        return super().fit_transform(X, y), y

    def transform(self, X, y=None):
        if not all([hasattr(self, attr) for attr in ['mean_', 'components_']]):
            self.fit(X, y)
        return super().transform(X), y
