from copy import deepcopy
from .nodes import nodefy, ListNode, IntParameterNode, FloatParameterNode, ChoiceNode, BooleanParameterNode


__all__ = [
    'hp_space',
    'hp_choice',
    'hp_int',
    'hp_boolean',
    'hp_float',
    'hp_list',
    'hp_space_from_json'
]


def hp_space(**values):
    assert len(values), 'missing argument values.'
    return nodefy(values)


def hp_choice(options, label=''):
    assert isinstance(options, list), 'options must be a list!'
    assert len(options) > 0, 'options cannot be empty!'
    return ChoiceNode(options, label=label)


def hp_int(min_value, max_value, label=''):
    assert min_value < max_value, 'max_value must be greater than min_value'
    return IntParameterNode(min_value, max_value, label=label)


def hp_float(min_value, max_value, label=''):
    assert min_value < max_value, 'max_value must be greater than min_value'
    return FloatParameterNode(min_value, max_value, label=label)


def hp_boolean(label=''):
    return BooleanParameterNode(label=label)


def hp_list(value, label=''):
    assert isinstance(value, list), "'value' must be a list"
    assert len(value) > 0, "'value' cannot be an empty list"
    return ListNode(value, label=label)


def hp_space_from_json(values):
    if isinstance(values, dict):
        class_name = values.get('class_name', '')
        config = deepcopy(values.get('config', {}))

        if class_name == 'ChoiceNode':
            config['value'] = [hp_space_from_json(v) for v in config['value']]
            return ChoiceNode(**config)

        elif class_name == 'ListNode':
            config['value'] = [hp_space_from_json(v) for v in config['value']]
            return ListNode(**config)

        elif class_name == 'IntParameterNode':
            return IntParameterNode(**config)

        elif class_name == 'FloatParameterNode':
            return FloatParameterNode(**config)

        elif class_name == 'BooleanParameterNode':
            return BooleanParameterNode(**config)

        else:
            vals = values.copy()
            for k, v in vals.items():
                vals[k] = hp_space_from_json(v)
            return nodefy(vals)

    if isinstance(values, list):
        return ListNode([hp_space_from_json(v) for v in values])

    return nodefy(values)
