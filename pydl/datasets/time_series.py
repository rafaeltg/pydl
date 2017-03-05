import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


def create_dataset(ts, look_back=1, time_ahead=1):
    """
    :param ts:
    :param look_back:
    :param time_ahead:
    :return:
    """

    assert len(ts) > look_back+time_ahead, 'No enough points in time series!'

    if isinstance(ts, pd.DataFrame):
        ts = ts.as_matrix()

    y_starts = range(look_back, len(ts)+1-time_ahead, 1)
    y_idxs = [range(i, i+time_ahead) for i in y_starts]
    x_idxs = [range(i-look_back, i) for i in y_starts]
    data_x, data_y = [], []
    for i in range(len(y_idxs)):
        data_x.append(ts[x_idxs[i], 0])
        data_y.append(np.reshape(ts[y_idxs[i], 0], time_ahead))
    return np.array(data_x), np.array(data_y)


def get_stock_historical_data(symbol, start, end, ascending=True, usecols=None):

    """
    :param symbol: stock ticker.
    :param start: string date in format 'yyyy-mm-dd' ('2009-09-11').
    :param end: string date in format 'yyyy-mm-dd' ('2010-09-11').
    :param ascending: sort returning values in ascending or descending order based on Date column.
    :param usecols: List of columns to return. If None, return all columns.
    :return: DataFrame
    """

    from yahoo_finance import Share

    stock = Share(symbol)
    data = stock.get_historical(start, end)

    df = pd.DataFrame(data).sort_values(by='Date', ascending=ascending)
    df = df.drop('Symbol', 1)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index(keys=['Date'], inplace=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])
    return df if usecols is None else df[usecols]


def get_return(x, periods=1):
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    return x.pct_change(periods) + 1


def get_log_return(x, periods=1):
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    return np.log(x).diff(periods).dropna()


def decompose(ts, plot=False):
    print(ts)
    decomposition = seasonal_decompose(ts, freq=7)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    if plot:
        plt.subplot(411)
        plt.plot(ts, label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(seasonal,label='Seasonality')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(residual, label='Residuals')
        plt.legend(loc='best')
        plt.tight_layout()

    return trend, seasonal, residual


def acf(ts, nlags=20, alpha=None, plot=False):
    from statsmodels.tsa.stattools import acf

    if alpha is not None:
        lag_acf, confint, qstat, pvalue = acf(ts, nlags=nlags, alpha=alpha,  qstat=True, unbiased=True)
    else:
        lag_acf, qstat, pvalue = acf(ts, nlags=nlags, qstat=True, unbiased=True)

    if plot:
        plt.plot(lag_acf)
        if alpha is not None:
            plt.plot(confint[:, 0], linestyle='--', color='red')
            plt.plot(confint[:, 1], linestyle='--', color='red')
        plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
        plt.title('Autocorrelation Function')
        plt.show(block=True)

    if alpha:
        return lag_acf, confint
    else:
        return lag_acf


def pacf(ts, nlags=20, method='ols', alpha=None, plot=False):
    from statsmodels.tsa.stattools import pacf

    if alpha is not None:
        lag_pacf, confint = pacf(ts, nlags=nlags, method=method, alpha=alpha)
    else:
        lag_pacf = pacf(ts, nlags=nlags, method=method)

    if plot:
        plt.plot(lag_pacf)
        plt.axhline(y=0,linestyle='--',color='gray')
        if alpha is not None:
            plt.plot(coefs[:, 0], linestyle='--', color='red')
            plt.plot(coefs[:, 1], linestyle='--', color='red')
        plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
        plt.title('Partial Autocorrelation Function')
        plt.show(block=True)

    if alpha:
        return lag_pacf, coefs
    else:
        return lag_pacf


def test_stationarity(ts):
    """
    Perform Dickey-Fuller test
    """

    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    return dfoutput


def smooth(ts, method='mean', window=5):

    if method == 'mean':
        return ts.rolling(window=window).mean()

    if method == 'ewma':
        return ts.ewm(span=window).mean()
