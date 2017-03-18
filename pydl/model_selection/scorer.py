from .metrics import *
from sklearn.metrics.scorer import *


# Regression scorers
mse_scorer = make_scorer(mse)
rmse_scorer = make_scorer(rmse)
mae_scorer = make_scorer(mae)
mape_scorer = make_scorer(mape)
neg_r2_scorer = make_scorer(r2_score, greater_is_better=False)

# Classification scorers
log_loss_scorer = make_scorer(log_loss)


_scorers = dict(
    mse=mse_scorer,
    rmse=rmse_scorer,
    mae=mae_scorer,
    mape=mape_scorer,
    r2=r2_scorer,
    neg_r2=neg_r2_scorer,
    accuracy=accuracy_scorer,
    f1=f1_scorer,
    log_loss=log_loss_scorer,
    precision=precision_scorer,
    recall=recall_scorer
)


def get_scorer(scorer):
    if isinstance(scorer, str):
        if scorer not in _scorers.keys():
            raise KeyError('%s is not a valid scorer!' % scorer)
        return _scorers[scorer]

    if hasattr(scorer, '__call__'):
        return scorer

    raise TypeError('Invalid scorer type!')
