from pydl.datasets import *
from pydl.model_selection import rmse, RegressionCV
from pydl.hyperopt import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)

look_back = 10
look_ahead = 1
time_steps = 1


def get_time_series():
    # S&P 500 daily log returns
    ts = load_csv('sp500_log_ret.csv', dtype={'Close': np.float64})

    # split into train and test sets
    train = ts['2000-01-01':'2016-03-09']
    test = ts['2016-03-10':'2017-03-10']

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x_train, y_train = create_dataset(train, look_back, look_ahead)
    x_test, y_test = create_dataset(test, look_back, look_ahead)

    x_train = np.reshape(x_train, (x_train.shape[0], time_steps, look_back))
    x_test = np.reshape(x_test, (x_test.shape[0], time_steps, look_back))

    return x_train, y_train, x_test, y_test


def run_opt(space, x, y):

    print('Creating Fitness Function')
    # Rolling window cv (10 years training, 1 year validation)
    fit_fn = CVObjectiveFunction(cv=RegressionCV('time_series', window=2772,  horizon=252, fixed=False, by=252))

    print('Creating CMAES optimizer')
    opt = CMAESOptimizer(pop_size=10, max_iter=10)

    print('Creating HyperOptModel...')
    m = HyperOptModel(hp_space=space, fit_fn=fit_fn, opt=opt)

    print('Optimizing!')
    res = m.fit(x, y, retrain=True, max_threads=4)

    return res


def run_sp500_lstm():

    print('Creating LSTM Hyperparameter Space')
    lstm_space = hp_space({
        'model': {
            'class_name': 'RNN',
            'config': hp_choice([
                {
                    'layers': [hp_int(10, 100)],
                    'cell_type': 'lstm',
                    'dropout': hp_float(0, 0.5),
                    'activation': hp_choice(['relu', 'tanh', 'sigmoid']),
                    'num_epochs': hp_int(100, 400),
                    'batch_size': hp_int(20, 300),
                    'opt': hp_choice(['rmsprop', 'adagrad', 'adam']),
                    'learning_rate': hp_float(0.00001, 0.001)
                },
                {
                    'layers': [hp_int(10, 100), hp_int(10, 100)],
                    'cell_type': 'lstm',
                    'dropout': [hp_float(0, 0.5), hp_float(0, 0.5)],
                    'activation': hp_choice(['relu', 'tanh', 'sigmoid']),
                    'num_epochs': hp_int(100, 400),
                    'batch_size': hp_int(20, 300),
                    'opt': hp_choice(['rmsprop', 'adagrad', 'adam']),
                    'learning_rate': hp_float(0.00001, 0.001)
                }
            ])
        }
    })

    x_train, y_train, x_test, y_test = get_time_series()

    # Let's do it!
    res = run_opt(lstm_space, x_train, y_train)

    print(res)


def plot_result(y_test, y_test_pred):
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


if __name__ == '__main__':
    run_sp500_lstm()
