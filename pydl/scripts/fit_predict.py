import os
from ..models.pipeline import Pipeline
from ..models.utils import save_json
from ..model_selection import available_metrics
from ..ts import acf, pacf, test_stationarity
from .fit import fit
from .predict import predict


def fit_predict(model,
                x_train,
                y_train,
                x_test=None,
                y_test=None,
                metrics: list = None,
                output_dir: str = '',
                save_model: bool = False,
                preds_save_format: str = None):

    """
    Fit and predict

    :param model:
    :param x_train: input data used to fit the model
    :param y_train: target values used to fit the model
    :param x_test: input data used for evaluating the built model
    :param y_test: target values used for evaluating the built model
    :param metrics: error metrics used to evaluate the model
    :param output_dir: output directory for the output files
    :param save_model:
    :param preds_save_format:
    """

    if not model.built:
        fit(
            model=model,
            x=x_train,
            y=y_train,
            save_built_model=save_model,
            output_dir=output_dir)

    y_train_pred = predict(
        model=model,
        x=x_train,
        filename='{}_y_train_pred'.format(model.name),
        save_format=preds_save_format,
        output_dir=output_dir)

    y_test_pred = None
    if x_test is not None and y_test is not None:
        y_test_pred = predict(
            model=model,
            x=x_test,
            filename='{}_y_test_pred'.format(model.name),
            save_format=preds_save_format,
            output_dir=output_dir)

    if metrics:
        if isinstance(model, Pipeline):
            _, y_train = model.transform(x_train, y_train)
            _, y_test = model.transform(x_test, y_test)

        result = {
            'train': pred_analysis(y_train, y_train_pred, metrics)
        }

        if y_test_pred is not None:
            result['test'] = pred_analysis(y_test, y_test_pred, metrics)

        save_json(result, os.path.join(output_dir, '{}_pred_metrics.json'.format(model.name)))


def pred_analysis(y_true, y_pred, metrics):
    ret = {m: available_metrics[m](y_true, y_pred) for m in metrics}

    up_true = (y_true > 0).as_type(int)
    up_pred = (y_pred > 0).as_type(int)
    ret['upward_precision'] = available_metrics['precision_score'](up_true, up_pred)
    ret['upward_recall'] = available_metrics['recall_score'](up_true, up_pred)

    down_true = (y_true < 0).as_type(int)
    down_pred = (y_pred < 0).as_type(int)
    ret['downward_precision'] = available_metrics['precision_score'](down_true, down_pred)
    ret['downward_recall'] = available_metrics['recall_score'](down_true, down_pred)

    residuals = y_true - y_pred
    acf_lags, acf_conf_level = acf(residuals, nlags=50)
    pacf_lags = pacf(residuals)
    adf = test_stationarity(residuals)

    ret.update({
        'residuals': {
            'acf': {
                'lags': acf_lags,
                'conf_level': acf_conf_level
            },
            'pacf': {
                'lags': pacf_lags.tolist()
            },
            'adf': {
                'test_statistic': adf['Test Statistic'],
                'p_value': adf['p-value'],
                'cv_1%': adf['Critical Value (1%)'],
                'cv_5%': adf['Critical Value (5%)'],
                'cv_10%': adf['Critical Value (10%)'],
            }
        }
    })

    return ret
