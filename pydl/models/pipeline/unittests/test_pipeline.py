import os
import unittest
import numpy as np
from pydl.models.utils import load_pipeline
from pydl.models.implementations import MLP
from pydl.models.pipeline import Pipeline, StandardScaler


class PipelineTestCase(unittest.TestCase):

    def test_init_invalid_steps(self):
        self.assertRaises(ValueError, Pipeline, steps=[])
        self.assertRaises(ValueError, Pipeline, steps=[1])
        self.assertRaises(ValueError, Pipeline, steps=[1, StandardScaler()])
        self.assertRaises(ValueError, Pipeline, steps=[StandardScaler()])  # missing estimator

    def test_init_valid_steps(self):
        p = Pipeline(steps=[MLP()])
        self.assertEqual(len(p), 1)
        self.assertListEqual(list(p.named_steps.keys()), ['mlp'])

        p = Pipeline(steps=['passthrough', MLP()])
        self.assertEqual(len(p), 2)
        self.assertListEqual(list(p.named_steps.keys()), ['mlp'])

        p = Pipeline(steps=[StandardScaler(), MLP()])
        self.assertEqual(len(p), 2)
        self.assertListEqual(list(p.named_steps.keys()), ['scaler', 'mlp'])

    def test_set_params(self):
        p = Pipeline(steps=[StandardScaler(), MLP()])
        self.assertEqual(p.named_steps['mlp'].out_activation, 'linear')

        p.set_params(mlp__out_activation='tanh')
        self.assertEqual(p.named_steps['mlp'].out_activation, 'tanh')

    def test_get_config(self):
        p = Pipeline(steps=[StandardScaler(), MLP()])
        config = p.get_config()

        self.assertListEqual(['name', 'steps'], list(config.keys()))
        self.assertEqual('pipeline', config['name'])
        self.assertEqual(2, len(config['steps']))
        self.assertEqual('StandardScaler', config['steps'][0]['class_name'])
        self.assertEqual('MLP', config['steps'][1]['class_name'])

    def test_get_config_with_passthrough(self):
        p = Pipeline(steps=[StandardScaler(), "passthrough", MLP()])
        config = p.get_config()

        self.assertListEqual(['name', 'steps'], list(config.keys()))
        self.assertEqual('pipeline', config['name'])
        self.assertEqual(2, len(config['steps']))
        self.assertEqual('StandardScaler', config['steps'][0]['class_name'])
        self.assertEqual('MLP', config['steps'][1]['class_name'])

    def test_from_config(self):
        config = {
            'name': 'pipeline',
            'steps': [
                {
                    'class_name': 'StandardScaler',
                    'config': {
                        'name': 'scaler',
                        'mean': [],
                        'std': []
                    }
                },
                {
                    'class_name': 'MLP',
                    'config': {
                        'name': 'MLP',
                        'layers': [],
                        'loss_func': 'mse', 'epochs': 1
                    }
                }
            ]
        }

        p = Pipeline.from_config(config=config)

        self.assertIsInstance(p, Pipeline)
        self.assertEqual(len(p), 2)
        self.assertListEqual(list(p.named_steps.keys()), ['scaler', 'mlp'])

    def test_from_config_with_passthrough(self):
        config = {
            'name': 'pipeline',
            'steps': [
                {
                    'class_name': 'StandardScaler',
                    'config': {
                        'name': 'scaler',
                        'mean': [],
                        'std': []
                    }
                },
                "passthrough",
                {
                    'class_name': 'MLP',
                    'config': {
                        'name': 'MLP',
                        'layers': [],
                        'loss_func': 'mse', 'epochs': 1
                    }
                }
            ]
        }

        p = Pipeline.from_config(config=config)

        self.assertIsInstance(p, Pipeline)
        self.assertEqual(len(p), 2)
        self.assertListEqual(list(p.named_steps.keys()), ['scaler', 'mlp'])

    def test_fit(self):
        p = Pipeline(steps=[StandardScaler(), MLP()])
        self.assertFalse(p.built)

        x, y = create_dataset()
        p.fit(x=x, y=y)
        self.assertTrue(p.built)

    def test_fit_with_passthrough(self):
        p = Pipeline(steps=[StandardScaler(), 'passthrough', MLP()])
        self.assertEqual(len(p), 3)
        self.assertListEqual(list(p.named_steps.keys()), ['scaler', 'mlp'])
        self.assertFalse(p.built)

        x, y = create_dataset()
        p.fit(x=x, y=y)
        self.assertTrue(p.built)

    def test_save_json(self):
        p = Pipeline(steps=[StandardScaler(), MLP()])
        self.assertFalse(p.built)

        x, y = create_dataset()
        p.fit(x=x, y=y)
        self.assertTrue(p.built)

        p.save_json()
        self.assertTrue(os.path.isfile('pipeline.json'))
        os.remove('pipeline.json')

    def test_save(self):
        p = Pipeline(steps=[StandardScaler(), MLP()])
        self.assertFalse(p.built)

        x, y = create_dataset()
        p.fit(x=x, y=y)
        self.assertTrue(p.built)

        p.save()
        self.assertTrue(os.path.isfile('pipeline.h5'))
        os.remove('pipeline.h5')

    def test_load_fom_h5_file(self):
        p = Pipeline(steps=[StandardScaler(), MLP()])
        self.assertFalse(p.built)

        x, y = create_dataset()
        p.fit(x=x, y=y)
        self.assertTrue(p.built)

        p.save()
        self.assertTrue(os.path.isfile('pipeline.h5'))

        p1 = load_pipeline('pipeline.h5')
        self.assertTrue(p1.built)
        self.assertIsInstance(p1.steps[0], StandardScaler)
        self.assertListEqual(list(p.steps[0].mean), list(p1.steps[0].mean))
        self.assertListEqual(list(p.steps[0].std), list(p1.steps[0].std))
        self.assertIsInstance(p1.steps[1], MLP)

        os.remove('pipeline.h5')


def create_dataset():
    x = [np.random.random_sample((10, 1)) for _ in range(5)]
    y = np.random.random_sample(10)
    return np.hstack(x), y
