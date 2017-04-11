import multiprocessing as mp
import numpy as np
from .methods import get_cv_method
from .scorer import get_scorer
from ..models.utils import model_from_config


class CV(object):

    """
        Cross-Validation
    """

    def __init__(self, method, **kwargs):
        self.cv = get_cv_method(method, **kwargs)

    def run(self, model, x, y=None, scoring=None, pp=None, max_threads=1):

        # get scorers
        if scoring is not None:
            if isinstance(scoring, list):
                scorers_fn = dict([(self.get_scorer_name(k), get_scorer(k)) for k in scoring])
            else:
                scorers_fn = dict([(self.get_scorer_name(scoring), get_scorer(scoring))])
        else:
            # By default uses the model loss function as scoring function
            scorers_fn = dict([(model.get_loss_func(), get_scorer(model.get_loss_func()))])

        model_cfg = model.to_json()

        args = []
        if y is not None:
            for train, test in self.cv.split(x, y):
                args.append((model_cfg,
                             x[train],
                             y[train],
                             x[test],
                             y[test],
                             scorers_fn,
                             pp))
            cv_results = self._do_cv(self._supervised_cv, args, max_threads)

        else:
            for train, test in self.cv.split(x):
                args.append((model_cfg,
                             x[train],
                             x[test],
                             scorers_fn))
            cv_results = self._do_cv(self._unsupervised_cv, args, max_threads)

        return self._consolidate_cv_scores(cv_results)

    def _do_cv(self, cv_fn, args, max_threads=1):
        if max_threads == 1:
            cv_results = [cv_fn(*fn_args) for fn_args in args]

        else:
            with mp.Pool(max_threads) as pool:
                cv_results = pool.starmap(func=cv_fn, iterable=args)

        return cv_results

    @staticmethod
    def _unsupervised_cv(model_cfg, x_train, x_test, scorers):
        model = model_from_config(model_cfg)
        model.fit(x_train=x_train)
        cv_result = dict([(k, scorer(model, x_test)) for k, scorer in scorers.items()])
        return cv_result

    @staticmethod
    def _supervised_cv(model_cfg, x_train, y_train, x_test, y_test, scorers, pp):
        model = model_from_config(model_cfg)

        # Data pre-processing
        if pp is not None:
            x_train = pp.fit_transform(x_train)
            x_test = pp.transform(x_test)

        model.fit(x_train, y_train)
        cv_result = dict([(k, scorer(model, x_test, y_test)) for k, scorer in scorers.items()])
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

    def get_scorer_name(self, scorer):
        if isinstance(scorer, str):
            return scorer
        elif hasattr(scorer, '__call__'):
            return scorer.__name__


