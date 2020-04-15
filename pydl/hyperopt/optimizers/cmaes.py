import cma
from .optimizer import Optimizer, _space_decoder_wrap, _parallel_objective
from ..nodes import Node


class CMAES(Optimizer):

    def __init__(self, max_iter=25, verbose=-9, **kwargs):
        """

        :param max_iter:
        :param verbose:
        :param pop_size:
        :param sigma0:
        :param bipop:
        :param adapt_sigma:
        :param verb_filenameprefix:
        """

        super().__init__(
            max_iter=max_iter,
            verbose=verbose,
            **kwargs)

        self.sigma0 = float(kwargs.get('sigma0', 0.5))
        assert self.sigma0 > 0, 'sigma0 must be greater than zero'

        self.bipop = bool(kwargs.get('bipop', False))
        self.adapt_sigma = bool(kwargs.get('adapt_sigma', False))
        self.verb_filenameprefix = kwargs.get('verb_filenameprefix', 'outcmaes')

    def fmin(self, search_space: Node, obj_func, args=(), x0=None, max_threads=1):
        assert max_threads > 0, 'Invalid number of threads!'

        if x0 is None:
            x0 = [.0] * search_space.size

        obj_func = _space_decoder_wrap(obj_func)

        opts = {
            'AdaptSigma': self.adapt_sigma,
            'maxiter': self.max_iter,
            'bounds': [0, 1],
            'popsize': self.pop_size,
            'ftarget': self.ftarget,
            'tolfun': self.tolfun,
            'verbose': self.verbose,
            'verb_filenameprefix': self.verb_filenameprefix
        }

        if max_threads == 1:
            result = cma.fmin(
                objective_function=obj_func,
                x0=x0,
                sigma0=self.sigma0,
                options=opts,
                args=(search_space, ) + args,
                bipop=self.bipop)

        else:
            result = cma.fmin(
                objective_function=None,
                parallel_objective=_parallel_objective(obj_func, max_threads),
                x0=x0,
                sigma0=self.sigma0,
                options=opts,
                args=(search_space, ) + args,
                bipop=self.bipop)

        return result[0]

    @classmethod
    def from_config(cls, config: dict):
        opt = super().from_config(config)
        opt.__dict__.update(config)
        return opt

    def get_config(self):
        config = super().get_config()
        config['sigma0'] = self.sigma0
        config['bipop'] = self.bipop
        config['adapt_sigma'] = self.adapt_sigma
        config['verb_filenameprefix'] = self.verb_filenameprefix
        return config
