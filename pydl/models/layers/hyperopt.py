from pydl.hyperopt.components import hp_space

"""
    Shortcuts hyperopt components
"""

def hp_layer(class_name: str, config: dict):
    return hp_space(
        class_name=class_name,
        config=hp_space(**config)
    )


def hp_dense(**kwargs):
    return hp_layer(
        class_name='Dense',
        config=kwargs
    )


def hp_dropout(**kwargs):
    return hp_layer(
        class_name='Dropout',
        config=kwargs
    )


def hp_spatial_dropout_1d(**kwargs):
    return hp_layer(
        class_name='SpatialDropout1D',
        config=kwargs
    )


def hp_lstm(**kwargs):
    return hp_layer(
        class_name='LSTM',
        config=kwargs
    )


def hp_gru(**kwargs):
    return hp_layer(
        class_name='GRU',
        config=kwargs
    )


def hp_cudnnlstm(**kwargs):
    return hp_layer(
        class_name='CuDNNLSTM',
        config=kwargs
    )


def hp_cudnngru(**kwargs):
    return hp_layer(
        class_name='CuDNNGRU',
        config=kwargs
    )


def hp_conv1d(**kwargs):
    return hp_layer(
        class_name='Conv1D',
        config=kwargs
    )


def hp_conv2d(**kwargs):
    return hp_layer(
        class_name='Conv2D',
        config=kwargs
    )


def hp_max_pooling_1d(**kwargs):
    return hp_layer(
        class_name='MaxPooling1D',
        config=kwargs
    )


def hp_convlstm1d(**kwargs):
    return hp_layer(
        class_name='ConvLSTM1D',
        config=kwargs
    )


def hp_convlstm2d(**kwargs):
    return hp_layer(
        class_name='ConvLSTM2D',
        config=kwargs
    )


def hp_flatten(**kwargs):
    return hp_layer(
        class_name='Flatten',
        config=kwargs
    )
