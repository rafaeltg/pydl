from .metrics import *
from sklearn.metrics.scorer import make_scorer, r2_scorer


# Regression scorers
mse_scorer = make_scorer(mse)
rmse_scorer = make_scorer(rmse)
mae_scorer = make_scorer(mae)
mape_scorer = make_scorer(mape)


_scorers = dict(
    mse=mse_scorer,
    rmse=rmse_scorer,
    mae=mae_scorer,
    mape=mape_scorer,
    r2=r2_scorer,
)


def get_scorer(scorer):
    if isinstance(scorer, str):
        if scorer not in _scorers.keys():
            raise KeyError('%s is not a valid scorer!' % scorer)
        return _scorers[scorer]

    if hasattr(scorer, '__call__'):
        return scorer

    raise TypeError('Invalid scorer type!')
