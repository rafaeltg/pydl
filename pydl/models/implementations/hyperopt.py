from pydl.hyperopt.components import hp_space


"""
    Shortcuts hyperopt components
"""


def hp_mlp(**kwargs):
    return hp_space(
        class_name='MLP',
        config=kwargs
    )


def hp_cnn(**kwargs):
    return hp_space(
        class_name='CNN',
        config=kwargs
    )


def hp_rnn(**kwargs):
    return hp_space(
        class_name='RNN',
        config=kwargs
    )


def hp_cnnlstm(**kwargs):
    return hp_space(
        class_name='CNNLSTM',
        config=kwargs
    )
