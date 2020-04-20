import numpy as np
from sklearn import metrics


mse = metrics.mean_squared_error
mae = metrics.mean_absolute_error
r2_score = metrics.r2_score


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    """
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, np.Inf)))


def corr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)


def upward_precision(y_true, y_pred):
    up_true = (y_true > 0).astype(int)
    up_pred = (y_pred > 0).astype(int)
    return metrics.precision_score(up_true, up_pred)


def upward_recall(y_true, y_pred):
    up_true = (y_true > 0).astype(int)
    up_pred = (y_pred > 0).astype(int)
    return metrics.recall_score(up_true, up_pred)


def downward_precision(y_true, y_pred):
    down_true = (y_true < 0).astype(int)
    down_pred = (y_pred < 0).astype(int)
    return metrics.precision_score(down_true, down_pred)


def downward_recall(y_true, y_pred):
    down_true = (y_true < 0).astype(int)
    down_pred = (y_pred < 0).astype(int)
    return metrics.recall_score(down_true, down_pred)


available_metrics = {
    'rmse': rmse,
    'mse': mse,
    'mae': mae,
    'mape': mape,
    'r2_score': r2_score,
    'variance': metrics.explained_variance_score,
    'upward_precision': upward_precision,
    'upward_recall': upward_recall,
    'downward_precision': downward_precision,
    'downward_recall': downward_recall,
    'accuracy': metrics.accuracy_score,
    'log_loss': metrics.log_loss,
    'recall_score': metrics.recall_score,
    'precision_score': metrics.precision_score
}
