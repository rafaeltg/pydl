import unittest
import numpy as np
from pydl.models.pipeline import Reshaper3D, Reshaper4D, Reshaper5D


class Reshaper3DTestCase(unittest.TestCase):

    def test_get_config(self):
        r = Reshaper3D(n_steps=5)
        config = r.get_config()

        expected_config = dict(
            n_steps=5
        )

        self.assertDictEqual(config, expected_config)

    def test_from_config(self):
        config = dict(
            n_steps=5
        )

        r = Reshaper3D.from_config(config=config)

        self.assertIsInstance(r, Reshaper3D)
        self.assertEqual(r.n_steps, 5)

    def test_to_json(self):
        r = Reshaper3D(n_steps=5)
        r_json = r.to_json()

        expected_json = b'{"class_name": "Reshaper3D", "config": {"n_steps": 5}}'
        self.assertEqual(r_json, expected_json)

    def test_reshape(self):
        r = Reshaper3D(n_steps=5)

        x, y = create_dataset()
        self.assertEqual(x.shape, (20, 5))
        self.assertEqual(y.shape, (20,))

        x, y = r.reshape(x, y)
        self.assertEqual(x.shape, (15, 5, 5))
        self.assertEqual(y.shape, (15,))


class Reshaper4DTestCase(unittest.TestCase):

    def test_get_config(self):
        r = Reshaper4D(n_steps=5, n_seqs=3)
        config = r.get_config()

        expected_config = dict(
            n_steps=5,
            n_seqs=3
        )

        self.assertDictEqual(config, expected_config)

    def test_from_config(self):
        config = dict(
            n_steps=5,
            n_seqs=3
        )

        r = Reshaper4D.from_config(config=config)

        self.assertIsInstance(r, Reshaper4D)
        self.assertEqual(r.n_steps, 5)
        self.assertEqual(r.n_seqs, 3)

    def test_to_json(self):
        r = Reshaper4D(n_steps=5, n_seqs=3)
        r_json = r.to_json()

        expected_json = b'{"class_name": "Reshaper4D", "config": {"n_steps": 5, "n_seqs": 3}}'
        self.assertEqual(r_json, expected_json)

    def test_reshape(self):
        r = Reshaper4D(n_steps=5, n_seqs=3)

        x, y = create_dataset()
        self.assertEqual(x.shape, (20, 5))
        self.assertEqual(y.shape, (20,))

        x, y = r.reshape(x, y)
        self.assertEqual(x.shape, (12, 3, 5, 5))
        self.assertEqual(y.shape, (12,))


class Reshaper5DTestCase(unittest.TestCase):

    def test_get_config(self):
        r = Reshaper5D(n_steps=5, n_seqs=3)
        config = r.get_config()

        expected_config = dict(
            n_steps=5,
            n_seqs=3
        )

        self.assertDictEqual(config, expected_config)

    def test_from_config(self):
        config = dict(
            n_steps=5,
            n_seqs=3
        )

        r = Reshaper5D.from_config(config=config)

        self.assertIsInstance(r, Reshaper5D)
        self.assertEqual(r.n_steps, 5)
        self.assertEqual(r.n_seqs, 3)

    def test_to_json(self):
        r = Reshaper5D(n_steps=5, n_seqs=3)
        r_json = r.to_json()

        expected_json = b'{"class_name": "Reshaper5D", "config": {"n_steps": 5, "n_seqs": 3}}'
        self.assertEqual(r_json, expected_json)

    def test_reshape(self):
        r = Reshaper5D(n_steps=5, n_seqs=3)

        x, y = create_dataset()
        self.assertEqual(x.shape, (20, 5))
        self.assertEqual(y.shape, (20,))

        x, y = r.reshape(x, y)
        self.assertEqual(x.shape, (12, 3, 1, 5, 5))
        self.assertEqual(y.shape, (12,))


def create_dataset():
    x = [np.random.random_sample((20, 1)) for _ in range(5)]
    y = np.random.random_sample(20)
    return np.hstack(x), y
