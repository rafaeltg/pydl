import multiprocessing as mp
import numpy as np

from .methods import get_cv_method
from .scorer import get_scorer


class CV(object):

    """
        Cross-Validation
    """

    def __init__(self, method, **kwargs):
        self.cv = get_cv_method(method, **kwargs)

    def run(self, model, x, y=None, scoring=None, pp=None, max_thread=1):

        # get scorers
        scorers_fn = {}
        if scoring is not None:
            if isinstance(scoring, list):
                scorers_fn.update([(k, get_scorer(k)) for k in scoring])
            else:
                scorers_fn.update((scoring, get_scorer(scoring)))

        args = []
        if y is not None:
            for train, test in self.cv.split(x, y):
                print('\n> train: [%s - %d]' % (train[0], train[-1]))
                print('> test: [%s - %d]' % (test[0], test[-1]))
                args.append((model.copy(),
                             x[train],
                             y[train],
                             x[test],
                             y[test],
                             scorers_fn,
                             pp))
            cv_results = self._do_cv(self._supervised_cv, args, max_thread)

        else:
            for train, test in self.cv.split(x):
                args.append((model.copy(),
                             x[train],
                             x[test],
                             scorers_fn))
            cv_results = self._do_cv(self._unsupervised_cv, args, max_thread)

        return self._consolidate_cv_scores(cv_results)

    def _do_cv(self, cv_fn, args, max_thread=1):
        cv_results = []
        if max_thread == 1:
            for fn_args in args:
                cv_results.append(cv_fn(*fn_args))

        else:
            with mp.Pool(max_thread) as pool:
                cv_results = pool.starmap(func=cv_fn, iterable=args)

        return cv_results

    @staticmethod
    def _unsupervised_cv(model, x_train, x_test, scorers):
        model.fit(x_train=x_train)

        cv_result = {model.get_loss_func(): model.score(x_test)}
        for k, scorer in scorers.items():
            cv_result[k] = scorer(model, x_test)

        return cv_result

    @staticmethod
    def _supervised_cv(model, x_train, y_train, x_test, y_test, scorers, pp=None):
        # Data pre-processing
        if pp:
            x_train = pp.fit_transform(x_train)
            x_test = pp.transform(x_test)

        model.fit(x_train, y_train)

        cv_result = {model.get_loss_func(): model.score(x_test, y_test)}
        for k, scorer in scorers.items():
            cv_result[k] = scorer(model, x_test, y_test)

        return cv_result

    def _consolidate_cv_scores(self, cv_results):
        cv_scores = {}
        for k in cv_results[0].keys():
            scores = [result[k] for result in cv_results]
            cv_scores[k] = {
                'values': scores,
                'mean': np.mean(scores),
                'sd': np.std(scores)
            }
        return cv_scores
