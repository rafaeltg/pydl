import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pydl.datasets.time_series import *
from pydl.datasets.synthetic import mackey_glass
from pydl.models import RNN
from pydl.model_selection import rmse, TimeSeriesCV, RegressionCV
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)

look_back = 10
look_ahead = 1
time_steps = 1


def run_sp500():
    sp500 = get_stock_historical_data('^GSPC', '2000-01-01', '2017-03-05', True, ['Close'])
    ts = get_log_return(sp500)
    ts = smooth(ts, method='ewma', window=5)

    test_stationarity(ts['Close'])

    # split into train and test sets
    train = ts['2000-01-01':'2016-03-04']
    test = ts['2016-03-05':'2017-03-05']

    # Let's do it!
    run_lstm(train, test)


def run_mackey_glass():
    ts = mackey_glass(sample_len=5000, seed=42)

    # Normalize the dataset
    ts = MinMaxScaler(feature_range=(0, 1)).fit_transform(ts)

    test_stationarity(ts)

    # split into train and test sets
    train_size = int(len(ts) * 0.9)
    train, test = ts[0:train_size], ts[train_size:len(ts)]

    # Let's do it!
    run_lstm(train, test)


def run_lstm(train, test):
    print('\n#Train: %d' % len(train))
    print('#Test: %d' % len(test))

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x_train, y_train = create_dataset(train, look_back, look_ahead)
    x_test, y_test = create_dataset(test, look_back, look_ahead)

    x_train = np.reshape(x_train, (x_train.shape[0], time_steps, look_back))
    x_test = np.reshape(x_test, (x_test.shape[0], time_steps, look_back))

    print('#x_train: %d' % len(x_train))
    print('#x_test: %d' % len(x_test))

    lstm = RNN(cell_type='lstm',
               layers=[50, 50],
               stateful=False,
               time_steps=time_steps,
               num_epochs=400,
               batch_size=200,
               opt='adam')

    print('\nTraining LSTM')
    lstm.fit(x_train, y_train)

    y_test_pred = lstm.predict(x_test)

    print('Test RMSE = %.4f' % rmse(y_test[:, 0], y_test_pred[:, 0]))
    print('Test corr = %.4f' % np.corrcoef(y_test[:, 0], y_test_pred[:, 0])[0, 1])

    errs = y_test[:, 0] - y_test_pred[:, 0]
    mean_err = np.average(errs)
    sd_err = np.std(errs)
    print('Test ME = %.4f' % mean_err)
    print('Test SdE = %.4f' % sd_err)

    gs = GridSpec(3, 2)

    # Actual x predicted
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(y_test[:, 0], label='Actual')
    ax1.plot(y_test_pred[:, 0], label='Pred')
    ax1.legend(loc='best')
    ax1.set_title('Test set predictions')

    # Residuals
    ax2 = plt.subplot(gs[1, :])
    ax2.plot(errs)
    ax2.axhline(y=mean_err, color='red', linestyle='-', label='Mean')
    ax2.axhline(y=mean_err+sd_err, color='green', linestyle='-', label='Mean +- std')
    ax2.axhline(y=mean_err-sd_err, color='green', linestyle='-')
    ax2.legend(loc='best')
    ax2.set_title('Residuals')

    # Residuals ACF
    ax3 = plt.subplot(gs[2, 0])
    ax3.plot(acf(errs))
    ax3.axhline(y=-1.96/np.sqrt(len(errs)), linestyle='--', color='gray')
    ax3.axhline(y=1.96/np.sqrt(len(errs)), linestyle='--', color='gray')
    ax3.set_title('Residuals Autocorrelation')

    # Residuals histogram
    ax4 = plt.subplot(gs[2, 1])
    ax4.hist(errs, 25, normed=1, facecolor='green', alpha=0.75)
    ax4.set_title('Residuals distribution')

    plt.show()


def run_forecast2():
    x, y = create_dataset(sp500, look_back, look_ahead)

    # split into train and test sets
    train_size = int(len(x) * 0.85)
    x_train, y_train = x[0:train_size], y[0:train_size]
    x_test, y_test = x[train_size:len(x)], y[train_size:len(y)]

    pp_x = StandardScaler()
    x_train = pp_x.fit_transform(x_train)
    x_test = pp_x.transform(x_test)

    pp_y = StandardScaler()
    y_train = pp_y.fit_transform(y_train)
    y_test = pp_y.transform(y_test)

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x_train = np.reshape(x_train, (x_train.shape[0], time_steps, look_back))
    x_test = np.reshape(x_test, (x_test.shape[0], time_steps, look_back))

    print('train = ', len(x_train))
    print('test = ', len(x_test))

    lstm = RNN(cell_type='lstm',
               layers=[50, 50],
               stateful=False,
               time_steps=time_steps,
               num_epochs=300,
               batch_size=200,
               opt='adam')

    print('Training LSTM')
    lstm.fit(x_train, y_train)
    y_test_pred = lstm.predict(x_test)

    y_test_pred = pp_y.inverse_transform(y_test_pred)
    y_test = pp_y.inverse_transform(y_test)

    test_pred = y_test_pred[range(0, len(y_test_pred), look_ahead)]
    test_pred = np.reshape(test_pred, (y_test.shape[0], 1))

    print(y_test.shape)
    print(test_pred.shape)
    print('Test MSE = %.3f' % mse(y_test.flatten(), test_pred.flatten()))
    print('Test corr = %.3f' % np.corrcoef(y_test.flatten(), test_pred.flatten())[0, 1])

    errs = y_test - test_pred
    mean_err = np.average(errs)
    sd_err = np.std(errs)
    print('Test ME = %.3f' % mean_err)
    print('Test SdE = %.3f' % sd_err)

    '''
    plt.figure(1)
    plt.plot(range(len(errs)), errs, label='Errors')
    plt.axhline(y=mean_err, color='red', linestyle='-')
    plt.axhline(y=mean_err+sd_err, color='green', linestyle='-')
    plt.axhline(y=mean_err-sd_err, color='green', linestyle='-')
    #plt.plot(range(len(test_pred)), test_pred, label='Pred')
    #plt.plot(range(len(test)), test, label='Actual')
    plt.legend(loc='best')
    plt.show()
    '''


def run_cv():

    print(len(sp500))

    lstm = RNN(cell_type='lstm',
               layers=[50, 50],
               stateful=False,
               time_steps=time_steps,
               num_epochs=300,
               batch_size=200,
               opt='adam')

    cv = TimeSeriesCV(window=1334, horizon=365, by=365, fixed=True, dataset_fn=create_dataset)

    print('Running CV!')
    res = cv.run(model=lstm, ts=sp500, metrics=['mape', 'rmse'], max_thread=2)

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))


def _create_dataset(x):
    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x, y = create_dataset(x, look_back, look_ahead)
    x = np.reshape(x, (x.shape[0], time_steps, look_back))
    return x, y


def run_cv1():

    lstm = RNN(cell_type='lstm',
               layers=[50, 50],
               stateful=False,
               time_steps=time_steps,
               num_epochs=300,
               batch_size=200,
               opt='adam')

    x, y = create_dataset(sp500, look_back, look_ahead)
    print(len(x))

    cv = RegressionCV(method='time_series', window=1319, horizon=365, by=365, fixed=True)

    print('Running CV!')
    res = cv.run(model=lstm, x=x, y=y, metrics=['mape', 'rmse'], max_thread=2)

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))

"""
def test():

    sp500_log_ret = get_log_return(sp500)
    x, y = create_dataset(sp500_log_ret, look_back, look_ahead)
    print(len(x))

    lstm = RNN(cell_type='lstm',
               layers=[50, 50],
               stateful=False,
               time_steps=time_steps,
               num_epochs=300,
               batch_size=200,
               opt='adam')

    cv = RegressionCV(method='time_series', window=1512, horizon=252, by=252, fixed=True)

    print('Running CV!')
    res = cv.run(model=lstm, x=x, y=y, metrics=['mape', 'rmse'], max_thread=2)

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))
"""

if __name__ == '__main__':
    run_sp500()
    #test()
    #run_cv()
    #run_cv1()
    #run_forecast2()