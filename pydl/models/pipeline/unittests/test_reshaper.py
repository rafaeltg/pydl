import unittest
import numpy as np
from pydl.models.pipeline import Reshaper3D, Reshaper4D, Reshaper5D


class Reshaper3DTestCase(unittest.TestCase):

    def test_get_config(self):
        r = Reshaper3D(n_steps=5)
        config = r.get_config()

        expected_config = dict(
            n_steps=5,
            name='reshaper'
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

        expected_json = '{"class_name": "Reshaper3D", "config": {"name": "reshaper", "n_steps": 5}}'
        self.assertEqual(expected_json, r_json)

    def test_set_params(self):
        r = Reshaper3D(n_steps=5)
        self.assertEqual(r.n_steps, 5)

        r.set_params(n_steps=6)
        self.assertEqual(r.n_steps, 6)

    def test_transform(self):
        r = Reshaper3D(n_steps=5)

        x, y = create_dataset()
        self.assertEqual(x.shape, (20, 5))
        self.assertEqual(y.shape, (20,))

        x, y = r.transform(x, y)
        self.assertEqual(x.shape, (15, 5, 5))
        self.assertEqual(y.shape, (15,))


class Reshaper4DTestCase(unittest.TestCase):

    def test_get_config(self):
        r = Reshaper4D(n_steps=5, n_seqs=3)
        config = r.get_config()

        expected_config = dict(
            n_steps=5,
            n_seqs=3,
            name='reshaper'
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

        expected_json = '{"class_name": "Reshaper4D", "config": {"name": "reshaper", "n_steps": 5, "n_seqs": 3}}'
        self.assertEqual(expected_json, r_json)

    def test_set_params(self):
        r = Reshaper4D(n_steps=5, n_seqs=4)
        self.assertEqual(r.n_steps, 5)
        self.assertEqual(r.n_seqs, 4)

        r.set_params(n_steps=6, n_seqs=5)
        self.assertEqual(r.n_steps, 6)
        self.assertEqual(r.n_seqs, 5)

    def test_transform(self):
        r = Reshaper4D(n_steps=5, n_seqs=3)

        x, y = create_dataset()
        self.assertEqual(x.shape, (20, 5))
        self.assertEqual(y.shape, (20,))

        x, y = r.transform(x, y)
        self.assertEqual(x.shape, (12, 3, 5, 5))
        self.assertEqual(y.shape, (12,))


class Reshaper5DTestCase(unittest.TestCase):

    def test_get_config(self):
        r = Reshaper5D(n_steps=5, n_seqs=3)
        config = r.get_config()

        expected_config = dict(
            n_steps=5,
            n_seqs=3,
            name='reshaper'
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

        expected_json = '{"class_name": "Reshaper5D", "config": {"name": "reshaper", "n_steps": 5, "n_seqs": 3}}'
        self.assertEqual(expected_json, r_json)

    def test_set_params(self):
        r = Reshaper5D(n_steps=5, n_seqs=4)
        self.assertEqual(r.n_steps, 5)
        self.assertEqual(r.n_seqs, 4)

        r.set_params(n_steps=6, n_seqs=5)
        self.assertEqual(r.n_steps, 6)
        self.assertEqual(r.n_seqs, 5)

    def test_transform(self):
        r = Reshaper5D(n_steps=5, n_seqs=3)

        x, y = create_dataset()
        self.assertEqual(x.shape, (20, 5))
        self.assertEqual(y.shape, (20,))

        x, y = r.transform(x, y)
        self.assertEqual(x.shape, (12, 3, 1, 5, 5))
        self.assertEqual(y.shape, (12,))


def create_dataset():
    x = [np.random.random_sample((20, 1)) for _ in range(5)]
    y = np.random.random_sample(20)
    return np.hstack(x), y
