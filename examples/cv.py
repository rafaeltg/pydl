import json
from pydl.models import MLP
from pydl.models.layers import Dense, Dropout
from pydl.model_selection import CV
from dataset import create_multivariate_data, train_test_split


def run_cv():

    """
        Cross-Validation examples
    """

    print('Creating dataset')
    n_features = 4
    data = create_multivariate_data(size=300, n_features=n_features)
    x, y, x_test, y_test = train_test_split(data[:, :-1], data[:, -1])

    print('Creating MLP')
    mlp = MLP(
        name='mlp',
        layers=[
            Dense(units=16, activation='relu'),
            Dropout(0.1),
            Dense(units=8, activation='relu')
        ],
        epochs=200
    )

    print('Creating TrainTestSplitCV method')
    cv = CV(method='split', test_size=0.2)

    print('Running CV!')
    res = cv.run(model=mlp, x=x, y=y, scoring=['rmse', 'r2'])

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))

    print('\nCreating TimeSeriesCV method')
    cv = CV(method='time_series', window=100, horizon=20, by=20, fixed=False)

    print('Running CV!')
    res = cv.run(model=mlp, x=x, y=y, scoring=['rmse', 'r2'], max_threads=1)

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))


if __name__ == '__main__':
    run_cv()
