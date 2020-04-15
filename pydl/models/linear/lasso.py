import numpy as np
import sklearn.linear_model as sk_lin
from .base import LinearMixin


class Lasso(sk_lin.Lasso, LinearMixin):

    def __init__(self, name='lasso', **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def get_config(self):
        config = {
            'name': self.name,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'fit_intercept': self.fit_intercept,
            'selection': self.selection
        }

        if self.built:
            config['coef_'] = self.coef_.flatten().tolist()
            config['intercept_'] = self.intercept_.flatten().tolist() if isinstance(self.intercept_, np.ndarray) else self.intercept_
            config['n_iter_'] = self.n_iter_

        return config

    @classmethod
    def from_config(cls, config: dict):
        coef = config.pop('coef_', None)
        if coef is not None:
            coef = np.asarray(coef)

        intercept = config.pop('intercept_', None)
        n_iter = config.pop('n_iter_', None)

        c = cls(**config)

        if coef is not None:
            c.__dict__['coef_'] = coef

        if intercept is not None:
            c.__dict__['intercept_'] = intercept

        if n_iter is not None:
            c.__dict__['n_iter_'] = n_iter

        return c

    @property
    def built(self):
        return hasattr(self, 'coef_')

