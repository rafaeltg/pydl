from .nodes import nodefy, ListNode, IntParameterNode, FloatParameterNode, ChoiceNode, BooleanParameterNode, \
    FeatureSelectionNode


def hp_space(**values):
    assert len(values), 'missing argument values.'
    return nodefy(values)


def hp_choice(options):
    assert isinstance(options, list), 'options must be a list!'
    assert len(options) > 0, 'options cannot be empty!'
    return ChoiceNode(options)


def hp_int(min_value, max_value):
    assert min_value < max_value, 'max_value must be greater than min_value'
    return IntParameterNode(min_value, max_value)


def hp_float(min_value, max_value):
    assert min_value < max_value, 'max_value must be greater than min_value'
    return FloatParameterNode(min_value, max_value)


def hp_boolean():
    return BooleanParameterNode()


def hp_list(value, label=''):
    assert isinstance(value, list), "'value' must be a list"
    assert len(value) > 0, "'value' cannot be an empty list"
    return ListNode(value, label)


def hp_feature_selection(n_features, label=''):
    assert n_features > 0, "'n_features' must be greater than zero"
    return FeatureSelectionNode(n_features, label=label)


def hp_model(class_name: str, config: dict):
    return hp_space(
        class_name=class_name,
        config=hp_space(**config)
    )


def hp_space_from_json(values):
    if isinstance(values, dict):
        if "node_type" in values:
            if values['node_type'] == 'hp_choice':
                return ChoiceNode([hp_space_from_json(v) for v in values['value']])

            elif values['node_type'] == 'hp_list':
                return ListNode([hp_space_from_json(v) for v in values['value']])

            elif values['node_type'] == 'hp_int':
                return IntParameterNode(values['min_val'], values['max_val'])

            elif values['node_type'] == 'hp_float':
                return FloatParameterNode(values['min_val'], values['max_val'])

            elif values['node_type'] == 'hp_boolean':
                return BooleanParameterNode()

            elif values['node_type'] == 'hp_feature_selection':
                return FeatureSelectionNode(values['value'])
        else:
            vals = values.copy()
            for k, v in vals.items():
                vals[k] = hp_space_from_json(v)
            return nodefy(vals)

    return nodefy(values)
