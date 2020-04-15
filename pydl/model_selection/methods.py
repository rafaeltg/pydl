from math import floor, ceil
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit


class TrainTestSplit:

    def __init__(self, test_size=0.3):
        self.test_size = test_size

    def split(self, X, y=None):
        assert len(X) > 0, 'X cannot be empty!'

        n = len(X)
        train_size = floor(n * (1 - self.test_size))
        yield list(range(0, train_size, 1)), list(range(train_size, n, 1))


class TimeSeriesSplit:

    def __init__(self,
                 window=None,
                 horizon=None,
                 n_folds=None,
                 window_size=0.8,
                 horizon_size=0.2,
                 fixed=True,
                 by=1):
        self.window = window
        self.horizon = horizon
        self.n_folds = n_folds
        self.window_size = window_size
        self.horizon_size = horizon_size
        self.fixed = fixed
        self.by = by

    def split(self, X, y=None):
        assert (X is not None) and (len(X) > 0), 'X cannot be empty!'

        if self.n_folds is not None:
            assert isinstance(self.window_size, float) and (self.window_size > 0), "'window_size' must be greater than zero"
            assert isinstance(self.horizon_size, float) and (self.horizon_size > 0), "'horizon_size' must be greater than zero"
            self._calc_params(len(X))

        if (self.window is not None) and (self.horizon is not None):
            assert len(X) >= (self.window+self.horizon), 'window size plus horizon size cannot be greater than input size!'

        starts_test = list(range(self.window, len(X)-self.horizon+1, self.by))

        if self.fixed:
            trains = [range(test_start-self.window, test_start) for test_start in starts_test]
        else:
            trains = [range(0, test_start) for test_start in starts_test]

        tests = [range(test_start, test_start+self.horizon) for test_start in starts_test]

        for i in range(0, len(trains)):
            yield list(trains[i]), list(tests[i])

    def _calc_params(self, x_size):
        aux = self.window_size + (self.n_folds * self.horizon_size)
        self.horizon = ceil(x_size * (self.horizon_size / aux))
        self.window = x_size - (self.n_folds * self.horizon)
        self.by = self.horizon


def get_cv_method(method, **kwargs):

    if method == 'kfold':
        return KFold(**kwargs)
    elif method == 'skfold':
        return StratifiedKFold(**kwargs)
    elif method == 'loo':
        return LeaveOneOut()
    elif method == 'shuffle_split':
        return ShuffleSplit(**kwargs)
    elif method == 'split':
        return TrainTestSplit(**kwargs)
    elif method == 's_shuffle_split':
        return StratifiedShuffleSplit(**kwargs)
    elif method == 'time_series':
        return TimeSeriesSplit(**kwargs)
    else:
        raise AttributeError('Invalid CV method - %s!' % method)
