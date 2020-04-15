from pydl.hyperopt.components import hp_space


"""
    Shortcuts hyperopt components
"""


def hp_pipeline(**kwargs):
    return hp_space(
        class_name='Pipeline',
        config=hp_space(**kwargs))


def hp_reshaper3d(n_steps):
    return hp_space(
        class_name='Reshaper3D',
        config=hp_space(n_steps=n_steps))


def hp_reshaper4d(n_steps, n_seqs):
    return hp_space(
        class_name='Reshaper4D',
        config=hp_space(n_steps=n_steps, n_seqs=n_seqs))


def hp_reshaper5d(n_steps, n_seqs):
    return hp_space(
        class_name='Reshaper5D',
        config=hp_space(n_steps=n_steps, n_seqs=n_seqs))
