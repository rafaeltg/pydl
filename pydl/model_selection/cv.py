import multiprocessing as mp
import numpy as np

from .methods import *
from .metrics import available_metrics
from ..datasets import create_dataset


class CV(object):

    """
        Cross-Validation
    """

    def __init__(self, method, **kwargs):
        self.cv = get_cv_method(method, **kwargs)

    def _get_metrics_map(self, metrics):
        metrics_fn = {}
        for m in metrics:
            assert m in available_metrics.keys(), 'Invalid metric - %s' % m
            metrics_fn[m] = available_metrics[m]
        return metrics_fn

    def _do_cv(self, cv_fn, args, max_thread=1):
        cv_results = []
        if max_thread == 1:
            for fn_args in args:
                cv_results.append(cv_fn(*fn_args))

        else:
            with mp.Pool(max_thread) as pool:
                cv_results = pool.starmap(func=cv_fn, iterable=args)

        return self._consolidate_cv_metrics(cv_results)

    def _consolidate_cv_metrics(self, cv_results):
        cv_metrics = {m: [] for m in cv_results[0].keys()}
        for result in cv_results:
            for k, v in result.items():
                cv_metrics[k].append(v)

        # Calculate mean and sd for each metric
        cv_result = {}
        for k, v in cv_metrics.items():
            cv_result[k] = {
                'values': v,
                'mean': np.mean(v),
                'sd': np.std(v)
            }

        return cv_result


class UnsupervisedCV(CV):

    def run(self, model, x, metrics=list([]), max_thread=1):

        """
        :param model:
        :param x:
        :param metrics:
        :param max_thread:
        :return:
        """

        metrics_fn = self._get_metrics_map(metrics)

        args = []
        for train, test in self.cv.split(x):
            args.append((model, x[train], x[test], metrics_fn))

        return self._do_cv(self._unsupervised_cv, args, max_thread)

    @staticmethod
    def _unsupervised_cv(model, x_train, x_test, metrics_fn):
        model.fit(x_train=x_train)

        cv_result = {model.get_loss_func(): model.score(x_test)}

        if len(metrics_fn) > 0:
            x_rec = model.reconstruct(model.transform(x_test))

            for k, fn in metrics_fn.items():
                cv_result[k] = fn(x_test, x_rec)

        return cv_result


class SupervisedCV(CV):

    def run(self, model, x, y=None, metrics=list([]), pp=None, max_thread=1):
        assert y is not None, 'Missing y!'

        metrics_fn = self._get_metrics_map(metrics)

        args = []
        for train, test in self.cv.split(x, y):
            print('\n> train: [%s - %d]' % (train[0], train[-1]))
            print('> test: [%s - %d]' % (test[0], test[-1]))
            args.append((model.copy(),
                         x[train],
                         y[train],
                         x[test],
                         y[test],
                         pp,
                         metrics_fn))

        return self._do_cv(self._cv_fn, args, max_thread)

    @staticmethod
    def _cv_fn(model, x_train, y_train, x_test, y_test, pp, metrics_fn):
        pass


class RegressionCV(SupervisedCV):

    @staticmethod
    def _cv_fn(model, x_train, y_train, x_test, y_test, pp, metrics_fn):

        # Data pre-processing
        if pp:
            x_train = pp.fit_transform(x_train)
            x_test = pp.transform(x_test)
            y_train = pp.fit_transform(y_train)
            y_test = pp.transform(y_test)

        model.fit(x_train=x_train, y_train=y_train)

        cv_result = {model.get_loss_func(): model.score(x_test, y_test)}

        if len(metrics_fn) > 0:
            y_pred = model.predict(x_test)

            if pp:
                y_test = pp.inverse_transform(y_test)
                y_pred = pp.inverse_transform(y_pred)

            for k, fn in metrics_fn.items():
                cv_result[k] = fn(y_test, y_pred)

        return cv_result


class ClassificationCV(SupervisedCV):

    @staticmethod
    def _cv_fn(model, x_train, y_train, x_test, y_test, pp, metrics_fn):

        # Data preprocessing
        if pp:
            x_train = pp.fit_transform(x_train)
            x_test = pp.transform(x_test)

        model.fit(x_train=x_train, y_train=y_train)

        cv_result = {model.get_loss_func(): model.score(x_test, y_test)}

        if len(metrics_fn) > 0:
            y_pred = model.predict(x_test)

            for k, fn in metrics_fn.items():
                if k == 'log_loss':
                    res = fn(y_test, model.predict(x_test, predic_probs=True))
                else:
                    res = fn(y_test, y_pred)

                cv_result[k] = res

        return cv_result


class TimeSeriesCV(CV):

    def __init__(self, window, horizon, by=1, fixed=False, dataset_fn=None):
        self.cv = get_cv_method('time_series', window=window, horizon=horizon, by=by, fixed=fixed)
        self.dataset_fn = dataset_fn if dataset_fn else create_dataset

    def run(self, model, ts, metrics=list([]), pp=None, max_thread=1):

        """
        :param model:
        :param ts:
        :param metrics:
        :param pp: Preprocessing object (sklearn.preprocessing)
        :param max_thread:
        :return:
        """

        metrics_fn = self._get_metrics_map(metrics)

        args = []
        for train, test in self.cv.split(ts):
            print('\n> train: [%s - %d]' % (train[0], train[-1]))
            print('> test: [%s - %d]' % (test[0], test[-1]))
            args.append((model.copy(),
                         ts[train],
                         ts[test],
                         pp,
                         metrics_fn,
                         self.dataset_fn))

        return self._do_cv(self._ts_cv, args, max_thread)

    @staticmethod
    def _ts_cv(model, train, test, pp, metrics_fn, dataset_fn):

        # Data pre-processing
        if pp:
            train = pp.fit_transform(train)
            test = pp.transform(test)

        x_train, y_train = dataset_fn(train)
        x_test, y_test = dataset_fn(test)

        model.fit(x_train=x_train, y_train=y_train)

        cv_result = {model.get_loss_func(): model.score(x_test, y_test)}

        if len(metrics_fn) > 0:
            y_pred = model.predict(x_test)

            if pp:
                y_pred = pp.inverse_transform(y_pred)
                y_test = pp.inverse_transform(y_test)

            for k, fn in metrics_fn.items():
                cv_result[k] = fn(y_test, y_pred)

        return cv_result
