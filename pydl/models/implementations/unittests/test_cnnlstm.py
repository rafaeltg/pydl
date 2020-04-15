import unittest
from pydl.models.implementations import CNNLSTM
from pydl.models.layers import Conv1D, MaxPooling1D, Dropout, Dense, LSTM, CuDNNLSTM, Flatten, TimeDistributed


class CNNLSTMTestCase(unittest.TestCase):

    def test_is_valid_layer(self):
        self.assertTrue(CNNLSTM.is_valid_layer(Conv1D(12, 3)), 'Conv1D is valid layer')
        self.assertTrue(CNNLSTM.is_valid_layer(MaxPooling1D()), 'MaxPooling1D is valid layer')
        self.assertTrue(CNNLSTM.is_valid_layer(LSTM(10)), 'LSTM is valid layer')
        self.assertTrue(CNNLSTM.is_valid_layer(CuDNNLSTM(10)), 'CuDNNLSTM is valid layer')
        self.assertTrue(CNNLSTM.is_valid_layer(Dropout(.1)), 'Dropout is valid layer')
        self.assertTrue(CNNLSTM.is_valid_layer(Flatten()), 'Flatten is valid layer')
        self.assertTrue(CNNLSTM.is_valid_layer(Dense(10)), 'Dense is valid layer')
        self.assertTrue(CNNLSTM.is_valid_layer(TimeDistributed(Conv1D(12, 3))),
                        'TimeDistributed(Conv1D) is valid layer')
        self.assertTrue(CNNLSTM.is_valid_layer(TimeDistributed(MaxPooling1D())),
                        'TimeDistributed(MaxPooling1D) is valid layer')
        self.assertTrue(CNNLSTM.is_valid_layer(TimeDistributed(Flatten())),
                        'TimeDistributed(Flatten) is valid layer')

    def test_cnnlstm_layers(self):
        layers_config = CNNLSTM.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            LSTM(units=16)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], TimeDistributed)
        self.assertIsInstance(layers_config[0].layer, Conv1D)
        self.assertIsInstance(layers_config[1], TimeDistributed)
        self.assertIsInstance(layers_config[1].layer, Flatten)
        self.assertIsInstance(layers_config[2], LSTM)

        layers_config = CNNLSTM.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            MaxPooling1D(pool_size=2),
            LSTM(units=5)
        ])
        self.assertEqual(4, len(layers_config))
        self.assertIsInstance(layers_config[0], TimeDistributed)
        self.assertIsInstance(layers_config[0].layer, Conv1D)
        self.assertIsInstance(layers_config[1], TimeDistributed)
        self.assertIsInstance(layers_config[1].layer, MaxPooling1D)
        self.assertIsInstance(layers_config[2], TimeDistributed)
        self.assertIsInstance(layers_config[2].layer, Flatten)
        self.assertIsInstance(layers_config[3], LSTM)

        layers_config = CNNLSTM.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            Conv1D(filters=32, kernel_size=3),
            LSTM(units=16)
        ])
        self.assertEqual(4, len(layers_config))
        self.assertIsInstance(layers_config[0], TimeDistributed)
        self.assertIsInstance(layers_config[0].layer, Conv1D)
        self.assertIsInstance(layers_config[1], TimeDistributed)
        self.assertIsInstance(layers_config[1].layer, Conv1D)
        self.assertIsInstance(layers_config[2], TimeDistributed)
        self.assertIsInstance(layers_config[2].layer, Flatten)
        self.assertIsInstance(layers_config[3], LSTM)

        layers_config = CNNLSTM.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            Conv1D(filters=32, kernel_size=3),
            MaxPooling1D(pool_size=2),
            LSTM(units=5)
        ])
        self.assertEqual(5, len(layers_config))
        self.assertIsInstance(layers_config[0], TimeDistributed)
        self.assertIsInstance(layers_config[0].layer, Conv1D)
        self.assertIsInstance(layers_config[1], TimeDistributed)
        self.assertIsInstance(layers_config[1].layer, Conv1D)
        self.assertIsInstance(layers_config[2], TimeDistributed)
        self.assertIsInstance(layers_config[2].layer, MaxPooling1D)
        self.assertIsInstance(layers_config[3], TimeDistributed)
        self.assertIsInstance(layers_config[3].layer, Flatten)
        self.assertIsInstance(layers_config[4], LSTM)

        layers_config = CNNLSTM.check_layers_config([
            Conv1D(filters=32, kernel_size=3),
            LSTM(units=5),
            Dense(units=10)
        ])
        self.assertEqual(4, len(layers_config))
        self.assertIsInstance(layers_config[0], TimeDistributed)
        self.assertIsInstance(layers_config[0].layer, Conv1D)
        self.assertIsInstance(layers_config[1], TimeDistributed)
        self.assertIsInstance(layers_config[1].layer, Flatten)
        self.assertIsInstance(layers_config[2], LSTM)
        self.assertIsInstance(layers_config[3], Dense)

        layers_config = CNNLSTM.check_layers_config([
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(units=16),
            Dense(units=10),
            Dropout(.1)
        ])
        self.assertEqual(5, len(layers_config))
        self.assertIsInstance(layers_config[0], TimeDistributed)
        self.assertIsInstance(layers_config[0].layer, Conv1D)
        self.assertIsInstance(layers_config[1], TimeDistributed)
        self.assertIsInstance(layers_config[1].layer, Flatten)
        self.assertIsInstance(layers_config[2], LSTM)
        self.assertIsInstance(layers_config[3], Dense)
        self.assertIsInstance(layers_config[4], Dropout)

    def test_cnnlstm_layers_from_config(self):
        model = CNNLSTM.from_config(
            config=dict(
                name='cnnlstm',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='LSTM',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], TimeDistributed)
        self.assertIsInstance(model.layers[0].layer, Conv1D)
        self.assertIsInstance(model.layers[1], TimeDistributed)
        self.assertIsInstance(model.layers[1].layer, Flatten)
        self.assertIsInstance(model.layers[2], LSTM)

        model = CNNLSTM.from_config(
            config=dict(
                name='cnnlstm',
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
                        class_name='LSTM',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(4, len(model.layers))
        self.assertIsInstance(model.layers[0], TimeDistributed)
        self.assertIsInstance(model.layers[0].layer, Conv1D)
        self.assertIsInstance(model.layers[1], TimeDistributed)
        self.assertIsInstance(model.layers[1].layer, MaxPooling1D)
        self.assertIsInstance(model.layers[2], TimeDistributed)
        self.assertIsInstance(model.layers[2].layer, Flatten)
        self.assertIsInstance(model.layers[3], LSTM)

        model = CNNLSTM.from_config(
            config=dict(
                name='cnnlstm',
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
                    ),
                    dict(
                        class_name='LSTM',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(4, len(model.layers))
        self.assertIsInstance(model.layers[0], TimeDistributed)
        self.assertIsInstance(model.layers[0].layer, Conv1D)
        self.assertIsInstance(model.layers[1], TimeDistributed)
        self.assertIsInstance(model.layers[1].layer, Conv1D)
        self.assertIsInstance(model.layers[2], TimeDistributed)
        self.assertIsInstance(model.layers[2].layer, Flatten)
        self.assertIsInstance(model.layers[3], LSTM)

        model = CNNLSTM.from_config(
            config=dict(
                name='cnnlstm',
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
                    ),
                    dict(
                        class_name='MaxPooling1D',
                        config=dict(
                            pool_size=2
                        )
                    ),
                    dict(
                        class_name='LSTM',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(5, len(model.layers))
        self.assertIsInstance(model.layers[0], TimeDistributed)
        self.assertIsInstance(model.layers[0].layer, Conv1D)
        self.assertIsInstance(model.layers[1], TimeDistributed)
        self.assertIsInstance(model.layers[1].layer, Conv1D)
        self.assertIsInstance(model.layers[2], TimeDistributed)
        self.assertIsInstance(model.layers[2].layer, MaxPooling1D)
        self.assertIsInstance(model.layers[3], TimeDistributed)
        self.assertIsInstance(model.layers[3].layer, Flatten)
        self.assertIsInstance(model.layers[4], LSTM)

        model = CNNLSTM.from_config(
            config=dict(
                name='cnnlstm',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='LSTM',
                        config=dict(
                            units=10
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
        self.assertIsInstance(model.layers[0], TimeDistributed)
        self.assertIsInstance(model.layers[0].layer, Conv1D)
        self.assertIsInstance(model.layers[1], TimeDistributed)
        self.assertIsInstance(model.layers[1].layer, Flatten)
        self.assertIsInstance(model.layers[2], LSTM)
        self.assertIsInstance(model.layers[3], Dense)

        model = CNNLSTM.from_config(
            config=dict(
                name='cnnlstm',
                layers=[
                    dict(
                        class_name='Conv1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='LSTM',
                        config=dict(
                            units=10
                        )
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
        self.assertEqual(5, len(model.layers))
        self.assertIsInstance(model.layers[0], TimeDistributed)
        self.assertIsInstance(model.layers[0].layer, Conv1D)
        self.assertIsInstance(model.layers[1], TimeDistributed)
        self.assertIsInstance(model.layers[1].layer, Flatten)
        self.assertIsInstance(model.layers[2], LSTM)
        self.assertIsInstance(model.layers[3], Dense)
        self.assertIsInstance(model.layers[4], Dropout)
