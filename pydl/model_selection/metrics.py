import numpy as np
from sklearn import metrics


mse = metrics.mean_squared_error
mae = metrics.mean_absolute_error


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


available_metrics = {
    'rmse': rmse,
    'mse': mse,
    'mae': mae,
    'mape': mape,
    'r2_score': metrics.r2_score,
    'variance': metrics.explained_variance_score,
    'accuracy': metrics.accuracy_score,
    'log_loss': metrics.log_loss
}
