import unittest
import numpy as np
from pydl.models.pipeline import PCA


class PCATestCase(unittest.TestCase):

    def test_get_config(self):
        s = PCA(n_components=5)
        config = s.get_config()

        expected_config = dict(
            n_components=5,
            whiten=False,
            name='pca'
        )

        self.assertDictEqual(config, expected_config)

    def test_from_config(self):
        config = dict(
            n_components=5,
            whiten=False,
            name='pca'
        )

        s = PCA.from_config(config=config)

        self.assertIsInstance(s, PCA)
        self.assertEqual(s.name, 'pca')
        self.assertEqual(s.n_components, 5)
        self.assertEqual(s.whiten, False)

    def test_to_json(self):
        s = PCA()
        s_json = s.to_json()

        expected_json = '{"class_name": "PCA", "config": {"n_components": null, "name": "pca", "whiten": false}}'
        self.assertEqual(expected_json, s_json)

    def test_fit(self):
        s = PCA(n_components=5)

        x, y = create_dataset()
        self.assertEqual((7, 10), x.shape)
        self.assertEqual((7,), y.shape)

        s.fit(x, y)
        self.assertTrue(s.components_ is not None)
        self.assertTrue(s.n_components_ is not None)
        self.assertTrue(s.mean_ is not None)

    def test_fit_transform(self):
        s = PCA(n_components=5)

        x, y = create_dataset()
        self.assertEqual((7, 10), x.shape)
        self.assertEqual((7,), y.shape)

        x, y = s.fit_transform(x, y)
        self.assertEqual((7, 5), x.shape)
        self.assertEqual((7,), y.shape)

    def test_transform_after_from_config(self):
        s = PCA(n_components=5)

        x, y = create_dataset()
        self.assertEqual((7, 10), x.shape)
        self.assertEqual((7,), y.shape)

        x1, y1 = s.fit_transform(x, y)

        config = s.get_config()
        self.assertIn('n_components', config)
        self.assertIn('whiten', config)
        self.assertIn('name', config)

        new_s = PCA.from_config(config)

        x2, y2 = new_s.transform(x, y)

        np.testing.assert_almost_equal(x1, x2, decimal=10)
        np.testing.assert_array_equal(y1, y2)


def create_dataset():
    x = [[[1.5], [2.0], [0.34], [-0.12], [0.67], [2.4], [-1.1]] for _ in range(10)]
    y = np.ones(7)
    return np.hstack(x), y
