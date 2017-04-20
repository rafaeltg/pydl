import cma
import multiprocessing as mp


class CMAESOptimizer:

    def __init__(self, pop_size=10, sigma0=0.5, max_iter=50, verbose=-9):
        assert pop_size > 0, 'pop_size must be greater than zero'
        assert sigma0 > 0 if isinstance(sigma0, float) else True, 'sigma0 must be greater than zero'
        assert max_iter > 0, 'max_iter must be greater than zero'

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.sigma0 = sigma0
        self.verbose = verbose
        self.child_initializer = None

    def optimize(self, x0, obj_func, args=(), max_threads=1):
        assert max_threads > 0, 'Invalid number of threads!'

        # TODO: 'AdaptSigma' option
        es = cma.CMAEvolutionStrategy(x0=x0,
                                      sigma0=self.sigma0,
                                      inopts={
                                          'maxiter': self.max_iter,
                                          'bounds': [0, 1],
                                          'popsize': self.pop_size,
                                          'verbose': self.verbose
                                      })

        if max_threads == 1:
            es.optimize(objective_fct=obj_func, args=args)
        else:
            max_threads = min(self.pop_size, max_threads)
            while not es.stop():
                X = es.ask()
                
                with mp.Pool(max_threads, initializer=self.child_initializer, initargs=args) as pool:
                    f_values = pool.map_async(obj_func, X, chunksize=es.popsize//max_threads).get()

                es.tell(X, f_values)
                es.disp()
                es.logger.add()

        return es.result()


def opt_from_config(algo, **kwargs):

    if algo == 'cmaes':
        return CMAESOptimizer(**kwargs)

    raise TypeError('Invalid optimizer algorithm (%s)' % algo)
