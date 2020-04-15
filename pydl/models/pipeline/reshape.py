import json
import numpy as np


__all__ = [
    'Reshaper3D',
    'Reshaper4D',
    'Reshaper5D'
]


def _split_sequences(x, y=None, n_steps=1):
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


class Reshaper:

    def __init__(self, n_steps):
        self.n_steps = n_steps

    def reshape(self, x, y=None):
        raise NotImplemented

    def get_config(self) -> dict:
        return {
            'n_steps': self.n_steps
        }

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def to_json(self, **kwargs) -> str:
        m = {
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }
        return json.dumps(m, **kwargs).encode('utf-8')


class Reshaper3D(Reshaper):

    def reshape(self, x, y=None):
        return _split_sequences(x, y, self.n_steps)


class Reshaper4D(Reshaper3D):
    def __init__(self, n_steps, n_seqs):
        super().__init__(n_steps=n_steps)
        self.n_seqs = n_seqs

    def reshape(self, x, y=None):
        x, y = super().reshape(x, y)
        return _split_sequences(x, y, self.n_seqs)

    def get_config(self) -> dict:
        c = super().get_config()
        c['n_seqs'] = self.n_seqs
        return c


class Reshaper5D(Reshaper4D):

    def reshape(self, x, y=None):
        x, y = _split_sequences(x, y, self.n_steps)
        x = x.reshape((-1, 1, self.n_steps, x.shape[-1]))
        return _split_sequences(x, y, self.n_seqs)
