import numpy as np
from joblib import Parallel, delayed
from ..nodes import Node


def _space_decoder_wrap(obj_func):
    def _wrap(x, space, *args):
        try:
            s = space.get_value(x)
            ret = obj_func(s, *args)
        except:
            ret = np.nan

        return ret

    return _wrap


def _parallel_objective(obj_func, max_threads):
    def _wrap(pop, *args):
        with Parallel(n_jobs=max_threads, batch_size=1) as parallel:
            f = parallel(delayed(function=obj_func, check_pickle=False)(x, *args) for x in pop)

        return f

    return _wrap


class Optimizer:

    def __init__(self, **kwargs):
        self.max_iter = int(kwargs.get('max_iter', 2))
        assert self.max_iter > 0, 'max_iter must be greater than zero'

        self.pop_size = int(kwargs.get('pop_size', 10))
        assert self.pop_size > 0, 'pop_size must be greater than zero'

        self.verbose = int(kwargs.get('verbose', -9))
        self.tolfun = float(kwargs.get('tolfun', 1e-11))
        self.ftarget = float(kwargs.get('ftarget', -1e-12))

    def fmin(self, search_space: Node, obj_func, args=(), x0=None, max_threads=1):
        pass

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def get_config(self):
        config = {
            'max_iter': self.max_iter,
            'tolfun': self.tolfun,
            'ftarget': self.ftarget,
            'verbose': self.verbose
        }

        return config
