import unittest
import numpy as np
from pydl.models.pipeline import StandardScaler


class StandardScalerTestCase(unittest.TestCase):

    def test_get_config(self):
        s = StandardScaler()
        config = s.get_config()

        expected_config = dict(
            name='scaler',
            mean=[],
            std=[]
        )

        self.assertDictEqual(config, expected_config)

    def test_from_config(self):
        config = dict(
            mean=[1, 2],
            std=[.1, .2]
        )

        s = StandardScaler.from_config(config=config)

        self.assertIsInstance(s, StandardScaler)
        self.assertEqual(s.name, 'scaler')
        self.assertEqual(s.mean, [1, 2])
        self.assertEqual(s.std, [.1, .2])

    def test_to_json(self):
        s = StandardScaler()
        s_json = s.to_json()

        expected_json = '{"class_name": "StandardScaler", "config": {"name": "scaler", "mean": [], "std": []}}'
        self.assertEqual(s_json, expected_json)

    def test_fit(self):
        s = StandardScaler()

        x, y = create_dataset()
        self.assertEqual(x.shape, (5, 3))
        self.assertEqual(y.shape, (5,))

        s.fit(x, y)
        np.testing.assert_array_almost_equal(list(s.mean), [3., 3., 3.])
        np.testing.assert_array_almost_equal(list(s.std), [1.414214, 1.414214, 1.414214])

    def test_fit_transform(self):
        s = StandardScaler()

        x, y = create_dataset()
        self.assertEqual(x.shape, (5, 3))
        self.assertEqual(y.shape, (5,))

        x, y = s.fit_transform(x, y)
        np.testing.assert_array_almost_equal(x, [
            [-1.414214, -1.414214, -1.414214],
            [-0.707107, -0.707107, -0.707107],
            [ 0.,        0.,        0.      ],
            [ 0.707107,  0.707107,  0.707107],
            [ 1.414214,  1.414214,  1.414214]])
        np.testing.assert_array_equal(y, [1, 1, 1, 1, 1])


def create_dataset():
    x = [[[1.], [2.], [3.], [4.], [5.]] for _ in range(3)]
    y = np.ones(5)
    return np.hstack(x), y
