import unittest
from pydl.models.implementations import CNN
from pydl.models.layers import Conv1D, Conv2D, MaxPooling1D, SpatialDropout1D, Dropout, Flatten, Dense, LSTM


class CNNTestCase(unittest.TestCase):

    def test_is_valid_layer(self):
        self.assertTrue(CNN.is_valid_layer(Conv1D(12, 3)), 'Conv1D is valid layer')
        self.assertTrue(CNN.is_valid_layer(Conv2D(12, (3, 3))), 'Conv2D is valid layer')
        self.assertTrue(CNN.is_valid_layer(MaxPooling1D()), 'MaxPooling1D is valid layer')
        self.assertTrue(CNN.is_valid_layer(SpatialDropout1D(.1)), 'SpatialDropout1D is valid layer')
        self.assertTrue(CNN.is_valid_layer(Dropout(.1)), 'Dropout is valid layer')
        self.assertTrue(CNN.is_valid_layer(Flatten()), 'Flatten is valid layer')
        self.assertTrue(CNN.is_valid_layer(Dense(10)), 'Dense is valid layer')
        self.assertFalse(CNN.is_valid_layer(LSTM(10)), 'LSTM is an invalid layer')

    def test_is_valid_conv_layer(self):
        self.assertTrue(CNN.is_valid_conv_layer(Conv1D(12, 3)), 'Conv1D is a valid conv layer')
        self.assertTrue(CNN.is_valid_layer(Conv2D(12, (3, 3))), 'Conv2D is a valid conv layer')
        self.assertTrue(CNN.is_valid_conv_layer(MaxPooling1D()), 'MaxPooling1D is a valid conv layer')
        self.assertTrue(CNN.is_valid_conv_layer(SpatialDropout1D(.1)), 'SpatialDropout1D is a valid conv layer')
        self.assertFalse(CNN.is_valid_conv_layer(Dropout(.1)), 'Dropout is an invalid conv layer')
        self.assertFalse(CNN.is_valid_conv_layer(Flatten()), 'Flatten is an invalid conv layer')
        self.assertFalse(CNN.is_valid_conv_layer(Dense(10)), 'Dense is an invalid conv layer')
        self.assertFalse(CNN.is_valid_conv_layer(LSTM(10)), 'LSTM is an invalid conv layer')

    def test_cnn_layers(self):
        layers_config = CNN.check_layers_config([
            Conv1D(filters=32, kernel_size=3)
        ])
        self.assertEqual(2, len(layers_config))
        self.assertIsInstance(layers_config[0], Conv1D)
        self.assertIsInstance(layers_config[1], Flatten)

        layers_config = CNN.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            Dense(units=5)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], Conv1D)
        self.assertIsInstance(layers_config[1], Flatten)
        self.assertIsInstance(layers_config[2], Dense)

        layers_config = CNN.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            Flatten(),
            Dense(units=5)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], Conv1D)
        self.assertIsInstance(layers_config[1], Flatten)
        self.assertIsInstance(layers_config[2], Dense)

        layers_config = CNN.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            Flatten(),
            Dense(units=5),
            Dropout(.2)
        ])
        self.assertEqual(4, len(layers_config))
        self.assertIsInstance(layers_config[0], Conv1D)
        self.assertIsInstance(layers_config[1], Flatten)
        self.assertIsInstance(layers_config[2], Dense)
        self.assertIsInstance(layers_config[3], Dropout)

        layers_config = CNN.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            Conv1D(filters=32, kernel_size=3)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], Conv1D)
        self.assertIsInstance(layers_config[1], Conv1D)
        self.assertIsInstance(layers_config[2], Flatten)

        layers_config = CNN.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            MaxPooling1D(pool_size=2)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], Conv1D)
        self.assertIsInstance(layers_config[1], MaxPooling1D)
        self.assertIsInstance(layers_config[2], Flatten)

        layers_config = CNN.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            MaxPooling1D(pool_size=2),
            Dense(12)
        ])
        self.assertEqual(4, len(layers_config))
        self.assertIsInstance(layers_config[0], Conv1D)
        self.assertIsInstance(layers_config[1], MaxPooling1D)
        self.assertIsInstance(layers_config[2], Flatten)
        self.assertIsInstance(layers_config[3], Dense)

        layers_config = CNN.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            SpatialDropout1D(0.1)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], Conv1D)
        self.assertIsInstance(layers_config[1], SpatialDropout1D)
        self.assertIsInstance(layers_config[2], Flatten)

        layers_config = CNN.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            SpatialDropout1D(0.1),
            MaxPooling1D(pool_size=2)
        ])
        self.assertEqual(4, len(layers_config))
        self.assertIsInstance(layers_config[0], Conv1D)
        self.assertIsInstance(layers_config[1], SpatialDropout1D)
        self.assertIsInstance(layers_config[2], MaxPooling1D)
        self.assertIsInstance(layers_config[3], Flatten)

    def test_cnn_layers_from_config(self):
        model = CNN.from_config(
            config=dict(
                name='cnn',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    )
                ]
            )
        )
        self.assertEqual(2, len(model.layers))
        self.assertIsInstance(model.layers[0], Conv1D)
        self.assertIsInstance(model.layers[1], Flatten)

        model = CNN.from_config(
            config=dict(
                name='cnn',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], Conv1D)
        self.assertIsInstance(model.layers[1], Flatten)
        self.assertIsInstance(model.layers[2], Dense)

        model = CNN.from_config(
            config=dict(
                name='cnn',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='Flatten',
                        config=dict()
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], Conv1D)
        self.assertIsInstance(model.layers[1], Flatten)
        self.assertIsInstance(model.layers[2], Dense)

        model = CNN.from_config(
            config=dict(
                name='cnn',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='Flatten',
                        config=dict()
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='Dropout',
                        config=dict(
                            rate=.1
                        )
                    )
                ]
            )
        )
        self.assertEqual(4, len(model.layers))
        self.assertIsInstance(model.layers[0], Conv1D)
        self.assertIsInstance(model.layers[1], Flatten)
        self.assertIsInstance(model.layers[2], Dense)
        self.assertIsInstance(model.layers[3], Dropout)

        model = CNN.from_config(
            config=dict(
                name='cnn',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], Conv1D)
        self.assertIsInstance(model.layers[1], Conv1D)
        self.assertIsInstance(model.layers[2], Flatten)

        model = CNN.from_config(
            config=dict(
                name='cnn',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='MaxPooling1D',
                        config=dict(
                            pool_size=2
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], Conv1D)
        self.assertIsInstance(model.layers[1], MaxPooling1D)
        self.assertIsInstance(model.layers[2], Flatten)

        model = CNN.from_config(
            config=dict(
                name='cnn',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='MaxPooling1D',
                        config=dict(
                            pool_size=2
                        )
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(4, len(model.layers))
        self.assertIsInstance(model.layers[0], Conv1D)
        self.assertIsInstance(model.layers[1], MaxPooling1D)
        self.assertIsInstance(model.layers[2], Flatten)
        self.assertIsInstance(model.layers[3], Dense)

        model = CNN.from_config(
            config=dict(
                name='cnn',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='SpatialDropout1D',
                        config=dict(
                            rate=.2
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], Conv1D)
        self.assertIsInstance(model.layers[1], SpatialDropout1D)
        self.assertIsInstance(model.layers[2], Flatten)

        model = CNN.from_config(
            config=dict(
                name='cnn',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='SpatialDropout1D',
                        config=dict(
                            rate=.1
                        )
                    ),
                    dict(
                        class_name='MaxPooling1D',
                        config=dict(
                            pool_size=2
                        )
                    )
                ]
            )
        )
        self.assertEqual(4, len(model.layers))
        self.assertIsInstance(model.layers[0], Conv1D)
        self.assertIsInstance(model.layers[1], SpatialDropout1D)
        self.assertIsInstance(model.layers[2], MaxPooling1D)
        self.assertIsInstance(model.layers[3], Flatten)
