import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stools
from statsmodels.tsa.seasonal import seasonal_decompose


__all__ = ['acf', 'pacf', 'test_stationarity', 'decompose', 'correlated_lags']


def acf(ts, nlags=20, plot=False, ax=None):

    """
    Autocorrelation function

    :param ts: time series
    :param nlags: number of lags to calculate the acf function
    :param plot: whether to plot the acf or not
    :param ax: custom plot axes
    :return:
        - acf value for each lag
        - confidence level value
        - plotted ax (if 'plot' is true)
    """

    lag_acf = stools.acf(ts, nlags=nlags)
    conf_level = 1.96/np.sqrt(len(ts))

    if plot:
        if ax is None:
            ax = plt.gca(xlim=(1, nlags), ylim=(-1.0, 1.0))

        ax.plot(lag_acf)
        ax.axhline(y=-conf_level, linestyle='--', color='gray')
        ax.axhline(y=conf_level, linestyle='--', color='gray')
        ax.set_title('Autocorrelation Function')
        ax.set_xlabel('Lags')
        ax.set_ylabel('ACF')
        return lag_acf, conf_level, ax

    return lag_acf.tolist(), conf_level


def pacf(ts, nlags=20, method='ols', alpha=None, plot=False, ax=None):

    """
    Partial autocorrelation function

    :param ts: time series
    :param nlags: number of lags to calculate the acf function
    :param method:
    :param alpha:
    :param plot: whether to plot the pacf or not
    :param ax: custom plot axes
    :return:
    """

    if alpha is not None:
        lag_pacf, confint = stools.pacf(ts, nlags=nlags, method=method, alpha=alpha)
    else:
        lag_pacf = stools.pacf(ts, nlags=nlags, method=method)

    if plot:
        if ax is None:
            ax = plt.gca(xlim=(1, nlags), ylim=(-1.0, 1.0))

        ax.plot(lag_pacf)
        ax.axhline(y=0, linestyle='--', color='gray')
        ax.set_title('Partial Autocorrelation Function')
        ax.set_xlabel('Lags')
        ax.set_ylabel('PACF')

        if alpha is not None:
            ax.plot(confint[:, 0], linestyle='--', color='red')
            ax.plot(confint[:, 1], linestyle='--', color='red')
            return lag_pacf, confint, ax
        else:
            conf_level = 1.96/np.sqrt(len(ts))
            ax.axhline(y=-conf_level, linestyle='--', color='gray')
            ax.axhline(y=conf_level, linestyle='--', color='gray')
            return lag_pacf, ax

    if alpha:
        return lag_pacf, confint
    else:
        return lag_pacf


def test_stationarity(ts, to_file='', log_result=False):
    """
    Perform Augmented Dickey-Fuller test
    """

    if isinstance(ts, np.ndarray):
        ts = ts.flatten()

    dftest = stools.adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for k, v in dftest[4].items():
        dfoutput['Critical Value (%s)' % k] = v

    if to_file != '':
        dfoutput.to_csv(to_file)
    elif log_result:
        print('Results of Dickey-Fuller Test:')
        print(dfoutput)

    return dfoutput


def decompose(ts, plot=False, axes=None):

    """
    Seasonal decomposition (Trend + Seasonality + Residual)

    :param ts: time series
    :param plot: whether to plot the seasonal components or not
    :param axes: custom list of plot axes
    :return:
    """

    decomposition = seasonal_decompose(ts, freq=7)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    if plot:
        if axes is None or not isinstance(axes, np.ndarray):
            _, axes = plt.subplots(4, 1, sharex=True)

        axes[0].plot(ts)
        axes[0].set_title('Original')
        axes[1].plot(trend)
        axes[1].set_title('Trend')
        axes[2].plot(seasonal)
        axes[2].set_title('Seasonality')
        axes[3].plot(residual)
        axes[3].set_title('Residuals')
        return trend, seasonal, residual, axes

    return trend, seasonal, residual


def correlated_lags(ts, corr_lags=1, max_lags=100):

    """
    Return the index of the correlated lags.

    :param ts: time series
    :param corr_lags: number of correlated lags to return. If -1, return all
    :param max_lags: number of lags to calculate the acf function
    """

    assert max_lags > corr_lags, "'max_lags' must be greater than 'corr_lags'"

    acfs, conf = acf(ts, max_lags)
    acfs = np.asarray(acfs)

    idx = np.argsort(acfs)

    most_corr = []

    for i in idx[-2::-1]:
        if acfs[i] > conf:
            most_corr.append(i)

        if len(most_corr) == corr_lags:
            break

    return sorted(most_corr)
