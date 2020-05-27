from pydl.hyperopt import ListNode, BooleanParameterNode, hp_space, hp_boolean

__all__ = [
    'hp_pipeline',
    'hp_reshaper3d',
    'hp_reshaper4d',
    'hp_reshaper5d',
    'hp_standard_scaler',
    'hp_base_filter_select',
    'hp_pca'
]


def hp_pipeline(steps, name='pipeline'):
    return hp_space(
        class_name='Pipeline',
        config=hp_space(steps=steps, name=name))


def hp_reshaper3d(n_steps, name='reshaper'):
    return hp_space(
        class_name='Reshaper3D',
        config=hp_space(n_steps=n_steps, name=name))


def hp_reshaper4d(n_steps, n_seqs, name='reshaper'):
    return hp_space(
        class_name='Reshaper4D',
        config=hp_space(n_steps=n_steps, n_seqs=n_seqs, name=name))


def hp_reshaper5d(n_steps, n_seqs, name='reshaper'):
    return hp_space(
        class_name='Reshaper5D',
        config=hp_space(n_steps=n_steps, n_seqs=n_seqs, name=name))


def hp_standard_scaler(name='scaler'):
    return hp_space(
        class_name='StandardScaler',
        config=hp_space(name=name))


def hp_base_filter_select(n_features, name='feature_selector'):
    return hp_space(
        class_name='BaseFilterSelect',
        config=hp_space(
            indexes=ListNode([BooleanParameterNode()] * n_features),
            name=name
        )
    )


def hp_pca(n_components, name='pca', **kwargs):
    return hp_space(
        class_name='PCA',
        config=hp_space(
            n_components=n_components,
            whiten=kwargs.get('whiten', hp_boolean(label='whiten')),
            name=name
        )
    )
