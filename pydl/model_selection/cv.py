import numpy as np
from joblib import Parallel, delayed
from .methods import get_cv_method
from .scorer import get_scorer
from ..models import Model, Pipeline, model_from_json


def _do_supervised_cv(model, train, test, x, y, scorers_fn):
    x_train, y_train = x[train], y[train]
    x_test, y_test = x[test], y[test]

    m = model_from_json(model) if isinstance(model, str) else model
    m.fit(x_train, y_train)
    if isinstance(m, Pipeline):
        _, y_test = m.transform(x_test, y_test)

    return dict([(k, scorer(m, x_test, y_test)) for k, scorer in scorers_fn.items()])


def _do_unsupervised_cv(model, train, test, x, scorers_fn):
    x_train = x[train]
    x_test = x[test]

    m = model_from_json(model) if isinstance(model, str) else model
    m.fit(x_train)
    return dict([(k, scorer(m, x_test)) for k, scorer in scorers_fn.items()])


class CV(object):

    """
        Cross-Validation
    """

    def __init__(self, method, **kwargs):
        self.cv = get_cv_method(method, **kwargs)

    def run(self, model, x, y=None, scoring=None, max_threads=1):

        # get scorers
        if scoring is not None:
            if isinstance(scoring, list):
                scorers_fn = dict([(self.get_scorer_name(k), get_scorer(k)) for k in scoring])
            else:
                scorers_fn = dict([(self.get_scorer_name(scoring), get_scorer(scoring))])
        else:
            # By default uses the model loss function as scoring function
            scorers_fn = dict([(model.get_loss_func(), get_scorer(model.get_loss_func()))])

        m = model.to_json() if isinstance(model, Model) else model

        if y is None:
            args = [(m, train, test, x, scorers_fn) for train, test in self.cv.split(x, y)]
            cv_fn = _do_unsupervised_cv
        else:
            args = [(m, train, test, x, y, scorers_fn) for train, test in self.cv.split(x, y)]
            cv_fn = _do_supervised_cv

        with Parallel(n_jobs=min(max_threads, len(args))) as parallel:
            cv_results = parallel(delayed(cv_fn)(*a) for a in args)

        return self._consolidate_cv_scores(cv_results)

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

    @staticmethod
    def get_scorer_name(scorer):
        if isinstance(scorer, str):
            return scorer
        elif hasattr(scorer, '__call__'):
            return scorer.__name__
