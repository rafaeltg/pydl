import numpy as np
from ._base_transformer import TransformerMixin, SerializerMixin

__all__ = [
    'Reshaper3D',
    'Reshaper4D',
    'Reshaper5D'
]


class Reshaper(TransformerMixin, SerializerMixin):

    def __init__(self, n_steps, name=None):
        self.n_steps = n_steps
        self.name = name or 'reshaper'

    def get_config(self) -> dict:
        return {
            'name': self.name,
            'n_steps': self.n_steps
        }

    def set_params(self, **params):
        if 'n_steps' in params:
            self.n_steps = params['n_steps']

        if 'name' in params:
            self.name = params['name']

    def _split_sequences(self, x, y=None, n_steps=1):
        if y is None:
            x_out = [x[i:(i + n_steps)] for i in range(len(x) - n_steps)]
            return np.array(x_out), None

        else:
            x_out, y_out = list(), list()

            for i in range(len(x) - n_steps):
                end_ix = i + n_steps
                x_out.append(x[i:end_ix])
                y_out.append(y[end_ix - 1])

            return np.array(x_out), np.array(y_out)


class Reshaper3D(Reshaper):

    def transform(self, X, y=None):
        return self._split_sequences(X, y, n_steps=self.n_steps)


class Reshaper4D(Reshaper3D):

    def __init__(self, n_steps, n_seqs, name=None):
        super().__init__(n_steps=n_steps, name=name)
        self.n_seqs = n_seqs

    def transform(self, X, y=None):
        X, y = super().transform(X, y)
        return self._split_sequences(X, y, n_steps=self.n_seqs)

    def set_params(self, **params):
        super().set_params(**params)

        if 'n_seqs' in params:
            self.n_seqs = params['n_seqs']

    def get_config(self) -> dict:
        c = super().get_config()
        c['n_seqs'] = self.n_seqs
        return c


class Reshaper5D(Reshaper4D):

    def transform(self, X, y=None):
        X, y = self._split_sequences(X, y, n_steps=self.n_steps)
        X = X.reshape((-1, 1, self.n_steps, X.shape[-1]))
        return self._split_sequences(X, y, self.n_seqs)
