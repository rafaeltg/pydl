import unittest
from pydl.models.implementations import RNN
from pydl.models.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU, Dropout, Dense, Flatten, ConvLSTM1D, ConvLSTM2D, Conv1D


class RNNTestCase(unittest.TestCase):

    def test_is_valid_layer(self):
        self.assertTrue(RNN.is_valid_layer(LSTM(10)), 'LSTM is a valid layer')
        self.assertTrue(RNN.is_valid_layer(CuDNNLSTM(10)), 'CuDNNLSTM is a valid layer')
        self.assertTrue(RNN.is_valid_layer(GRU(10)), 'GRU is a valid layer')
        self.assertTrue(RNN.is_valid_layer(CuDNNGRU(10)), 'CuDNNGRU is a valid layer')
        self.assertTrue(RNN.is_valid_layer(ConvLSTM1D(10, 3)), 'ConvLSTM1D is a valid layer')
        self.assertTrue(RNN.is_valid_layer(ConvLSTM2D(10, (3, 3))), 'ConvLSTM2D is a valid layer')
        self.assertTrue(RNN.is_valid_layer(Dense(10)), 'Dense is a valid layer')
        self.assertTrue(RNN.is_valid_layer(Dropout(0.1)), 'Dropout is a valid layer')
        self.assertTrue(RNN.is_valid_layer(Flatten()), 'Flatten is a valid layer')
        self.assertFalse(RNN.is_valid_layer(Conv1D(10, 10)), 'Conv1D is an invalid layer')

    def test_lstm_layers(self):
        layers_config = RNN.check_layers_config([
            LSTM(units=10)
        ])
        self.assertEqual(1, len(layers_config))
        self.assertIsInstance(layers_config[0], LSTM)
        self.assertFalse(layers_config[0].return_sequences)

        layers_config = RNN.check_layers_config([
            LSTM(units=10),
            Dense(units=5)
        ])
        self.assertEqual(2, len(layers_config))
        self.assertIsInstance(layers_config[0], LSTM)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Dense)

        layers_config = RNN.check_layers_config([
            LSTM(units=10),
            Dense(units=5),
            Dropout(0.1)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], LSTM)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Dense)
        self.assertIsInstance(layers_config[2], Dropout)

        layers_config = RNN.check_layers_config([
            LSTM(units=10),
            LSTM(units=10)
        ])
        self.assertEqual(2, len(layers_config))
        self.assertIsInstance(layers_config[0], LSTM)
        self.assertTrue(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], LSTM)
        self.assertFalse(layers_config[1].return_sequences)

    def test_lstm_layers_from_config(self):
        model = RNN.from_config(
            config=dict(
                name='lstm',
                layers=[
                    dict(
                        class_name='LSTM',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(1, len(model.layers))
        self.assertIsInstance(model.layers[0], LSTM)
        self.assertFalse(model.layers[0].return_sequences)

        model = RNN.from_config(
            config=dict(
                name='lstm',
                layers=[
                    dict(
                        class_name='LSTM',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=5
                        )
                    )
                ]
            )
        )
        self.assertEqual(2, len(model.layers))
        self.assertIsInstance(model.layers[0], LSTM)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Dense)

        model = RNN.from_config(
            config=dict(
                name='lstm',
                layers=[
                    dict(
                        class_name='LSTM',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=5
                        )
                    ),
                    dict(
                        class_name='Dropout',
                        config=dict(
                            rate=0.1
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], LSTM)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Dense)
        self.assertIsInstance(model.layers[2], Dropout)

        model = RNN.from_config(
            config=dict(
                name='lstm',
                layers=[
                    dict(
                        class_name='LSTM',
                        config=dict(
                            units=10
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
        self.assertEqual(2, len(model.layers))
        self.assertIsInstance(model.layers[0], LSTM)
        self.assertTrue(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], LSTM)
        self.assertFalse(model.layers[1].return_sequences)

    def test_cudnnlstm_layers(self):
        layers_config = RNN.check_layers_config([
            CuDNNLSTM(units=10)
        ])
        self.assertEqual(1, len(layers_config))
        self.assertIsInstance(layers_config[0], CuDNNLSTM)
        self.assertFalse(layers_config[0].return_sequences)

        layers_config = RNN.check_layers_config([
            CuDNNLSTM(units=10),
            Dense(units=5)
        ])
        self.assertEqual(2, len(layers_config))
        self.assertIsInstance(layers_config[0], CuDNNLSTM)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Dense)

        layers_config = RNN.check_layers_config([
            CuDNNLSTM(units=10),
            Dense(units=5),
            Dropout(0.1)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], CuDNNLSTM)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Dense)
        self.assertIsInstance(layers_config[2], Dropout)

        layers_config = RNN.check_layers_config([
            CuDNNLSTM(units=10),
            CuDNNLSTM(units=10)
        ])
        self.assertEqual(2, len(layers_config))
        self.assertIsInstance(layers_config[0], CuDNNLSTM)
        self.assertTrue(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], CuDNNLSTM)
        self.assertFalse(layers_config[1].return_sequences)

    def test_cudnnlstm_layers_from_config(self):
        model = RNN.from_config(
            config=dict(
                name='cudnnlstm',
                layers=[
                    dict(
                        class_name='CuDNNLSTM',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(1, len(model.layers))
        self.assertIsInstance(model.layers[0], CuDNNLSTM)
        self.assertFalse(model.layers[0].return_sequences)

        model = RNN.from_config(
            config=dict(
                name='lscudnnlstmtm',
                layers=[
                    dict(
                        class_name='CuDNNLSTM',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=5
                        )
                    )
                ]
            )
        )
        self.assertEqual(2, len(model.layers))
        self.assertIsInstance(model.layers[0], CuDNNLSTM)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Dense)

        model = RNN.from_config(
            config=dict(
                name='cudnnlstm',
                layers=[
                    dict(
                        class_name='CuDNNLSTM',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=5
                        )
                    ),
                    dict(
                        class_name='Dropout',
                        config=dict(
                            rate=0.1
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], CuDNNLSTM)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Dense)
        self.assertIsInstance(model.layers[2], Dropout)

        model = RNN.from_config(
            config=dict(
                name='cudnnlstm',
                layers=[
                    dict(
                        class_name='CuDNNLSTM',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='CuDNNLSTM',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(2, len(model.layers))
        self.assertIsInstance(model.layers[0], CuDNNLSTM)
        self.assertTrue(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], CuDNNLSTM)
        self.assertFalse(model.layers[1].return_sequences)

    def test_gru_layers(self):
        layers_config = RNN.check_layers_config([
            GRU(units=10)
        ])
        self.assertEqual(1, len(layers_config))
        self.assertIsInstance(layers_config[0], GRU)
        self.assertFalse(layers_config[0].return_sequences)

        layers_config = RNN.check_layers_config([
            GRU(units=10),
            Dense(units=5)
        ])
        self.assertEqual(2, len(layers_config))
        self.assertIsInstance(layers_config[0], GRU)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Dense)

        layers_config = RNN.check_layers_config([
            GRU(units=10),
            Dense(units=5),
            Dropout(0.1)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], GRU)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Dense)
        self.assertIsInstance(layers_config[2], Dropout)

        layers_config = RNN.check_layers_config([
            GRU(units=10),
            GRU(units=10)
        ])
        self.assertEqual(2, len(layers_config))
        self.assertIsInstance(layers_config[0], GRU)
        self.assertTrue(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], GRU)
        self.assertFalse(layers_config[1].return_sequences)

    def test_gru_layers_from_config(self):
        model = RNN.from_config(
            config=dict(
                name='gru',
                layers=[
                    dict(
                        class_name='GRU',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(1, len(model.layers))
        self.assertIsInstance(model.layers[0], GRU)
        self.assertFalse(model.layers[0].return_sequences)

        model = RNN.from_config(
            config=dict(
                name='gru',
                layers=[
                    dict(
                        class_name='GRU',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=5
                        )
                    )
                ]
            )
        )
        self.assertEqual(2, len(model.layers))
        self.assertIsInstance(model.layers[0], GRU)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Dense)

        model = RNN.from_config(
            config=dict(
                name='gru',
                layers=[
                    dict(
                        class_name='GRU',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=5
                        )
                    ),
                    dict(
                        class_name='Dropout',
                        config=dict(
                            rate=0.1
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], GRU)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Dense)
        self.assertIsInstance(model.layers[2], Dropout)

        model = RNN.from_config(
            config=dict(
                name='gru',
                layers=[
                    dict(
                        class_name='GRU',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='GRU',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(2, len(model.layers))
        self.assertIsInstance(model.layers[0], GRU)
        self.assertTrue(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], GRU)
        self.assertFalse(model.layers[1].return_sequences)

    def test_cudnngru_layers(self):
        layers_config = RNN.check_layers_config([
            CuDNNGRU(units=10)
        ])
        self.assertEqual(1, len(layers_config))
        self.assertIsInstance(layers_config[0], CuDNNGRU)
        self.assertFalse(layers_config[0].return_sequences)

        layers_config = RNN.check_layers_config([
            CuDNNGRU(units=10),
            Dense(units=5)
        ])
        self.assertEqual(2, len(layers_config))
        self.assertIsInstance(layers_config[0], CuDNNGRU)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Dense)

        layers_config = RNN.check_layers_config([
            CuDNNGRU(units=10),
            Dense(units=5),
            Dropout(0.1)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], CuDNNGRU)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Dense)
        self.assertIsInstance(layers_config[2], Dropout)

        layers_config = RNN.check_layers_config([
            CuDNNGRU(units=10),
            CuDNNGRU(units=10)
        ])
        self.assertEqual(2, len(layers_config))
        self.assertIsInstance(layers_config[0], CuDNNGRU)
        self.assertTrue(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], CuDNNGRU)
        self.assertFalse(layers_config[1].return_sequences)

    def test_cudnngru_layers_from_config(self):
        model = RNN.from_config(
            config=dict(
                name='cudnngru',
                layers=[
                    dict(
                        class_name='CuDNNGRU',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(1, len(model.layers))
        self.assertIsInstance(model.layers[0], CuDNNGRU)
        self.assertFalse(model.layers[0].return_sequences)

        model = RNN.from_config(
            config=dict(
                name='cudnngru',
                layers=[
                    dict(
                        class_name='CuDNNGRU',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=5
                        )
                    )
                ]
            )
        )
        self.assertEqual(2, len(model.layers))
        self.assertIsInstance(model.layers[0], CuDNNGRU)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Dense)

        model = RNN.from_config(
            config=dict(
                name='cudnngru',
                layers=[
                    dict(
                        class_name='CuDNNGRU',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='Dense',
                        config=dict(
                            units=5
                        )
                    ),
                    dict(
                        class_name='Dropout',
                        config=dict(
                            rate=0.1
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], CuDNNGRU)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Dense)
        self.assertIsInstance(model.layers[2], Dropout)

        model = RNN.from_config(
            config=dict(
                name='cudnngru',
                layers=[
                    dict(
                        class_name='CuDNNGRU',
                        config=dict(
                            units=10
                        )
                    ),
                    dict(
                        class_name='CuDNNGRU',
                        config=dict(
                            units=10
                        )
                    )
                ]
            )
        )
        self.assertEqual(2, len(model.layers))
        self.assertIsInstance(model.layers[0], CuDNNGRU)
        self.assertTrue(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], CuDNNGRU)
        self.assertFalse(model.layers[1].return_sequences)

    def test_convlstm_layers(self):
        layers_config = RNN.check_layers_config([
            ConvLSTM1D(filters=32, kernel_size=3)
        ])
        self.assertEqual(2, len(layers_config))
        self.assertIsInstance(layers_config[0], ConvLSTM1D)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Flatten)

        layers_config = RNN.check_layers_config([
            ConvLSTM1D(filters=32, kernel_size=3),
            Dense(10)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], ConvLSTM1D)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Flatten)
        self.assertIsInstance(layers_config[2], Dense)

        layers_config = RNN.check_layers_config([
            ConvLSTM1D(filters=32, kernel_size=3),
            Flatten(),
            Dense(10)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], ConvLSTM1D)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Flatten)
        self.assertIsInstance(layers_config[2], Dense)

        layers_config = RNN.check_layers_config([
            ConvLSTM1D(filters=32, kernel_size=3),
            Dropout(0.1)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], ConvLSTM1D)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Flatten)
        self.assertIsInstance(layers_config[2], Dropout)

        layers_config = RNN.check_layers_config([
            ConvLSTM1D(filters=32, kernel_size=3),
            Flatten(),
            Dropout(0.1)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], ConvLSTM1D)
        self.assertFalse(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], Flatten)
        self.assertIsInstance(layers_config[2], Dropout)

        layers_config = RNN.check_layers_config([
            ConvLSTM1D(filters=32, kernel_size=3),
            ConvLSTM1D(filters=32, kernel_size=3)
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], ConvLSTM1D)
        self.assertTrue(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], ConvLSTM1D)
        self.assertFalse(layers_config[1].return_sequences)
        self.assertIsInstance(layers_config[2], Flatten)

        layers_config = RNN.check_layers_config([
            ConvLSTM1D(filters=32, kernel_size=3),
            ConvLSTM1D(filters=32, kernel_size=3),
            Flatten()
        ])
        self.assertEqual(3, len(layers_config))
        self.assertIsInstance(layers_config[0], ConvLSTM1D)
        self.assertTrue(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], ConvLSTM1D)
        self.assertFalse(layers_config[1].return_sequences)
        self.assertIsInstance(layers_config[2], Flatten)

        layers_config = RNN.check_layers_config([
            ConvLSTM1D(filters=32, kernel_size=3),
            ConvLSTM1D(filters=32, kernel_size=3),
            Dense(10)
        ])
        self.assertEqual(4, len(layers_config))
        self.assertIsInstance(layers_config[0], ConvLSTM1D)
        self.assertTrue(layers_config[0].return_sequences)
        self.assertIsInstance(layers_config[1], ConvLSTM1D)
        self.assertFalse(layers_config[1].return_sequences)
        self.assertIsInstance(layers_config[2], Flatten)
        self.assertIsInstance(layers_config[3], Dense)

    def test_convlstm_layers_from_config(self):
        model = RNN.from_config(
            config=dict(
                name='convlstm',
                layers=[
                    dict(
                        class_name='ConvLSTM1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    )
                ]
            )
        )
        self.assertEqual(2, len(model.layers))
        self.assertIsInstance(model.layers[0], ConvLSTM1D)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Flatten)

        model = RNN.from_config(
            config=dict(
                name='convlstm',
                layers=[
                    dict(
                        class_name='ConvLSTM1D',
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
            ),
            custom_objects={'ConvLSTM1D': ConvLSTM1D}
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], ConvLSTM1D)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Flatten)
        self.assertIsInstance(model.layers[2], Dense)

        model = RNN.from_config(
            config=dict(
                name='convlstm',
                layers=[
                    dict(
                        class_name='ConvLSTM1D',
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
            ),
            custom_objects={'ConvLSTM1D': ConvLSTM1D}
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], ConvLSTM1D)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Flatten)
        self.assertIsInstance(model.layers[2], Dense)

        model = RNN.from_config(
            config=dict(
                name='convlstm',
                layers=[
                    dict(
                        class_name='ConvLSTM1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='Dropout',
                        config=dict(
                            rate=.1
                        )
                    )
                ]
            ),
            custom_objects={'ConvLSTM1D': ConvLSTM1D}
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], ConvLSTM1D)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Flatten)
        self.assertIsInstance(model.layers[2], Dropout)

        model = RNN.from_config(
            config=dict(
                name='convlstm',
                layers=[
                    dict(
                        class_name='ConvLSTM1D',
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
                        class_name='Dropout',
                        config=dict(
                            rate=.1
                        )
                    )
                ]
            ),
            custom_objects={'ConvLSTM1D': ConvLSTM1D}
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], ConvLSTM1D)
        self.assertFalse(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], Flatten)
        self.assertIsInstance(model.layers[2], Dropout)

        model = RNN.from_config(
            config=dict(
                name='convlstm',
                layers=[
                    dict(
                        class_name='ConvLSTM1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='ConvLSTM1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], ConvLSTM1D)
        self.assertTrue(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], ConvLSTM1D)
        self.assertFalse(model.layers[1].return_sequences)
        self.assertIsInstance(model.layers[2], Flatten)

        model = RNN.from_config(
            config=dict(
                name='convlstm',
                layers=[
                    dict(
                        class_name='ConvLSTM1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='ConvLSTM1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='Flatten',
                        config=dict()
                    )
                ]
            )
        )
        self.assertEqual(3, len(model.layers))
        self.assertIsInstance(model.layers[0], ConvLSTM1D)
        self.assertTrue(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], ConvLSTM1D)
        self.assertFalse(model.layers[1].return_sequences)
        self.assertIsInstance(model.layers[2], Flatten)

        model = RNN.from_config(
            config=dict(
                name='convlstm',
                layers=[
                    dict(
                        class_name='ConvLSTM1D',
                        config=dict(
                            filters=32,
                            kernel_size=3
                        )
                    ),
                    dict(
                        class_name='ConvLSTM1D',
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
        self.assertEqual(4, len(model.layers))
        self.assertIsInstance(model.layers[0], ConvLSTM1D)
        self.assertTrue(model.layers[0].return_sequences)
        self.assertIsInstance(model.layers[1], ConvLSTM1D)
        self.assertFalse(model.layers[1].return_sequences)
        self.assertIsInstance(model.layers[2], Flatten)
        self.assertIsInstance(model.layers[3], Dense)
