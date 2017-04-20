import multiprocessing as mp
import numpy as np
from .methods import get_cv_method
from .scorer import get_scorer
from ..models.utils import model_from_config


def _supervised_cv(x_train, y_train, x_test, y_test, model_config, scorers, pp):
    model = model_from_config(model_config)

    # Data pre-processing
    if pp is not None:
        x_train = pp.fit_transform(x_train)
        x_test = pp.transform(x_test)

    model.fit(x_train, y_train)
    cv_result = dict([(k, scorer(model, x_test, y_test)) for k, scorer in scorers.items()])
    return cv_result


def _supervised_cv_parallel(train_idxs, test_idxs):
    x_train, y_train = x[train_idxs], y[train_idxs]
    x_test, y_test   = x[test_idxs], y[test_idxs]

    model = model_from_config(model_cfg)

    # Data pre-processing
    if pp is not None:
        x_train = pp.fit_transform(x_train)
        x_test = pp.transform(x_test)

    model.fit(x_train, y_train)
    cv_result = dict([(k, scorer(model, x_test, y_test)) for k, scorer in scorers_fn.items()])
    return cv_result


def _unsupervised_cv(train_idxs, test_idxs):
    x_train = x[train_idxs]
    x_test = x[test_idxs]

    model = model_from_config(model_cfg)
    model.fit(x_train=x_train)
    cv_result = dict([(k, scorer(model, x_test)) for k, scorer in scorers_fn.items()])
    return cv_result


def _child_initialize(_model_cfg, _x, _y, _scorers_fn, _pp):
    global model_cfg, x, y, scorers_fn, pp
    model_cfg = _model_cfg
    x = _x
    y = _y
    scorers_fn = _scorers_fn
    pp = _pp


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
        args = [(train, test) for train, test in self.cv.split(x, y)]

        if max_threads == 1:
            if y is None:
                cv_results = [_unsupervised_cv(x[arg[0]], x[arg[1]], model_cfg, scorers_fn) for arg in args]
            else:
                cv_results = [_supervised_cv(x[arg[0]], y[arg[0]], x[arg[1]], y[arg[1]], model_cfg, scorers_fn, pp) for arg in args]

        else:
            cv_fn = _supervised_cv_parallel if y is not None else _unsupervised_cv
            max_threads = min(max_threads, len(args))
            with mp.Pool(max_threads, initializer=_child_initialize, initargs=(model_cfg, x, y, scorers_fn, pp)) as pool:
                cv_results = pool.starmap(func=cv_fn, iterable=args, chunksize=len(args)//max_threads)

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

    def get_scorer_name(self, scorer):
        if isinstance(scorer, str):
            return scorer
        elif hasattr(scorer, '__call__'):
            return scorer.__name__


