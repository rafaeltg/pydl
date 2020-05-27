import unittest
from pydl.hyperopt.components import *


class ComponentsTestCase(unittest.TestCase):

    def test_hp_choice(self):
        self.assertRaises(AssertionError, hp_choice, 1)
        self.assertRaises(AssertionError, hp_choice, [])

        x = hp_choice([1, 2, 3])
        self.assertEqual(x.size, 1)
        self.assertEqual(x.get_value([0.5]), 2)

        a = hp_int(1, 4)
        b = hp_int(1, 4)
        x = hp_choice([a, b])
        self.assertEqual(x.size, 2)
        self.assertEqual(x.get_value([0, 0]), 1)

        x = hp_choice([hp_choice([a, b]), a, 'lala'])
        self.assertEqual(x.size, 3)
        self.assertEqual(x.get_value([1, 1, 0]), 'lala')

    def test_hp_int(self):
        min_v = 1
        max_v = 10

        self.assertRaises(AssertionError, hp_int, max_v, min_v)

        p = hp_int(min_v, max_v)

        self.assertEqual(p.size, 1)
        self.assertEqual(min_v, p.get_value([0]))
        self.assertEqual(max_v, p.get_value([1]))
        self.assertEqual(5, p.get_value([0.5]))

    def test_hp_float(self):
        min_v = 0
        max_v = 1

        self.assertRaises(AssertionError, hp_float, max_v, min_v)

        p = hp_float(min_v, max_v)

        self.assertEqual(p.size, 1)
        self.assertEqual(min_v, p.get_value([0]))
        self.assertEqual(max_v, p.get_value([1]))
        self.assertEqual(0.5, p.get_value([0.5]))

    def test_hp_space(self):
        self.assertRaises(AssertionError, hp_space)

        space = hp_space(a=1, b=2)
        self.assertEqual(space.size, 0)

        expected = {
            'a': 1,
            'b': 2
        }

        self.assertDictEqual(expected, space.get_value([0, 0.5, 0, 1]))

    def test_hp_space_from_json(self):
        hp_space_json = {
            "class_name": "MLP",
            "config": {
                "name": "mlp_opt",
                "layers": {
                    "class_name": "ListNode",
                    "config": {
                        "value": [
                            {
                                "class_name": "IntParameterNode",
                                "config": {
                                    "min_val": 10,
                                    "max_val": 100
                                }
                            },
                            {
                                "class_name": "IntParameterNode",
                                "config": {
                                    "min_val": 10,
                                    "max_val": 100
                                }
                            }
                        ]
                    }
                },
                "dropout": {
                    "class_name": "FloatParameterNode",
                    "config": {
                        "min_val": 0,
                        "max_val": .3
                    }
                }
            }
        }

        space = hp_space_from_json(hp_space_json)

        self.assertEqual(space.size, 3)

        expected_config = {
            'class_name': 'MLP',
            'config': {
                'dropout': 0.0,
                'name': 'mlp_opt',
                'layers': [55, 100]
            }
        }

        self.assertDictEqual(expected_config, space.get_value([0, 0.5, 1]))

    def test_hp_space_to_json(self):
        expected_json = {
            "class_name": "MLP",
            "config": {
                "name": "mlp_opt",
                "layers": {
                    "class_name": "ListNode",
                    "config": {
                        "value": [
                            {
                                "class_name": "IntParameterNode",
                                "config": {
                                    "min_val": 10,
                                    "max_val": 100,
                                    "label": ''
                                }
                            },
                            {
                                "class_name": "IntParameterNode",
                                "config": {
                                    "min_val": 10,
                                    "max_val": 100,
                                    "label": ''
                                }
                            }
                        ],
                        "label": "layers"
                    }
                },
                "dropout": {
                    "class_name": "FloatParameterNode",
                    "config": {
                        "min_val": 0,
                        "max_val": .3,
                        "label": "dropout"
                    }
                }
            }
        }

        space = hp_space_from_json(expected_json)

        actual_json = space.to_json()

        self.assertDictEqual(actual_json, expected_json)


if __name__ == '__main__':
    unittest.main()
