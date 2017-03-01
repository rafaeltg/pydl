import numpy as np
from sklearn import metrics


def rmse(y_true, y_pred):
    """ Root Mean Squared Error	"""

    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    """ Mean Absolute Percentage Error """

    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, np.Inf)))


mse = metrics.mean_squared_error
mae = metrics.mean_absolute_error

available_metrics = {
    'rmse': rmse,
    'mse': mse,
    'mae': mae,
    'mape': mape,
    'variance': metrics.explained_variance_score,
    'accuracy': metrics.accuracy_score,
    'log_loss': metrics.log_loss
}
