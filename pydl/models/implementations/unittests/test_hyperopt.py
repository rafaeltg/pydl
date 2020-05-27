import unittest
from pydl.hyperopt import hp_choice, hp_int
from pydl.models.implementations import hp_mlp


class HyperoptComponentsTestCase(unittest.TestCase):

    def test_hp_mlp(self):
        space = hp_mlp(
            activation=hp_choice(['relu', 'sigmoid', 'tanh']),
            layers=[hp_int(10, 100), hp_int(10, 100)]
        )

        self.assertEqual(space.size, 3)

        expected_config = {
            'class_name': 'MLP',
            'config': {
                'activation': 'relu',
                'layers': [55, 100]
            }
        }

        self.assertDictEqual(expected_config, space.get_value([0, 0.5, 1]))
