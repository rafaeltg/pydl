import unittest
import numpy as np
from pydl.models.pipeline import BaseFilterSelect


class BaseFilterSelectTestCase(unittest.TestCase):

    def test_get_config(self):
        r = BaseFilterSelect(indexes=[0, 1])
        config = r.get_config()

        expected_config = dict(
            indexes=[0, 1],
            name='feature_selector'
        )

        self.assertDictEqual(config, expected_config)

    def test_from_config(self):
        config = dict(
            indexes=[0, 1]
        )

        r = BaseFilterSelect.from_config(config=config)

        self.assertIsInstance(r, BaseFilterSelect)
        self.assertEqual(r.indexes, [0, 1])

    def test_to_json(self):
        r = BaseFilterSelect(indexes=[0, 1])
        r_json = r.to_json()

        expected_json = '{"class_name": "BaseFilterSelect", "config": {"indexes": [0, 1], "name": "feature_selector"}}'
        self.assertEqual(r_json, expected_json)

    def test_set_params(self):
        s = BaseFilterSelect(indexes=[0, 1])
        self.assertEqual(s.indexes, [0, 1])

        s.set_params(indexes=[2, 3])
        self.assertEqual(s.indexes, [2, 3])

    def test_transform(self):
        r = BaseFilterSelect(indexes=[0, 1])

        x, y = create_dataset()
        self.assertEqual(x.shape, (10, 5))
        self.assertEqual(y.shape, (10,))

        x, y = r.transform(x, y)
        self.assertEqual(x.shape, (10, 2))
        self.assertEqual(y.shape, (10,))


def create_dataset():
    x = [np.random.random_sample((10, 1)) for _ in range(5)]
    y = np.random.random_sample(10)
    return np.hstack(x), y
