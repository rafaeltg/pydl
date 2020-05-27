import math as m


def _calc_size(space):
    size = 0
    if isinstance(space, list) and len(space) > 0:
        size = sum([_calc_size(v) for v in space])

    elif isinstance(space, Node):
        return space.size

    return size


class Node:
    def __init__(self, value, label=''):
        self._label = label
        self._value = value
        self._size = self._get_size()

    def _get_size(self):
        return _calc_size(self._value)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    def get_value(self, x):
        assert len(x) >= self._size, 'x must contains at least %d items' % self._size

        if isinstance(self._value, list):
            ret_params = {}
            param_idx = 0
            for v in self._value:
                ret_params[v.label] = v.get_value(x[param_idx:(param_idx + v.size)])
                param_idx += v.size
            return ret_params

        # literals (i.e. int, float, string...)
        return self._value

    def to_json(self):
        if isinstance(self._value, list):
            return dict([(v.label, v.to_json()) for v in self._value])

        return self._value

    def get_config(self) -> dict:
        return {
            "value": [v.to_json() for v in self._value] if isinstance(self._value, list) else self._value,
            "label": self.label
        }


class ChoiceNode(Node):
    def __init__(self, value, label=''):
        super().__init__(value=[nodefy(v) for v in value], label=label)

    def _get_size(self):
        return max([v.size for v in self._value]) + 1

    def get_value(self, x):
        n = self._value[int(round(x[0] * (len(self._value)-1)))]
        return n.get_value(x[1:])

    def to_json(self):
        return {
            "class_name": self.__class__.__name__,
            "config": self.get_config()
        }

    def get_config(self) -> dict:
        return {
            "value": [v.to_json() for v in self._value],
            "label": self.label
        }


class IntParameterNode(Node):
    def __init__(self, min_val, max_val, label=''):
        super().__init__(value=None, label=label)
        self._min = min_val
        self._max = max_val

    def _get_size(self):
        return 1

    def get_value(self, x):
        return m.floor(self._min + x[0] * (self._max - self._min))

    def to_json(self):
        return {
            "class_name": self.__class__.__name__,
            "config": self.get_config()
        }

    def get_config(self) -> dict:
        return {
            "min_val": self._min,
            "max_val": self._max,
            "label": self.label
        }


class FloatParameterNode(Node):
    def __init__(self, min_val, max_val, label=''):
        super().__init__(None, label)
        self._min = min_val
        self._max = max_val

    def _get_size(self):
        return 1

    def get_value(self, x):
        return self._min + x[0] * (self._max - self._min)

    def to_json(self):
        return {
            "class_name": self.__class__.__name__,
            "config": self.get_config()
        }

    def get_config(self) -> dict:
        return {
            "min_val": self._min,
            "max_val": self._max,
            "label": self.label
        }


class BooleanParameterNode(Node):
    def __init__(self, label=''):
        super().__init__(value=None, label=label)

    def _get_size(self):
        return 1

    def get_value(self, x):
        return bool((x[0] if isinstance(x, list) else x) > .5)

    def to_json(self):
        return {
            "class_name": self.__class__.__name__,
            "config": self.get_config()
        }

    def get_config(self) -> dict:
        return {
            "label": self.label
        }


class ListNode(Node):
    def __init__(self, value, label=''):
        super().__init__(value, label)

    def _get_size(self):
        return sum([v.size for v in self._value])

    def get_value(self, x):
        ret_params = []
        param_idx = 0
        for v in self._value:
            v_size = v.size
            ret_params.append(v.get_value(x[param_idx:(param_idx+v_size)]))
            param_idx += v_size
        return ret_params

    def to_json(self):
        return {
            "class_name": self.__class__.__name__,
            "config": self.get_config()
        }

    def get_config(self) -> dict:
        return {
            "value": [v.to_json() for v in self._value],
            "label": self.label
        }


def nodefy(value) -> Node:
    if isinstance(value, dict) and len(value) > 0:
        values = []
        for k, v in value.items():
            n = nodefy(v)
            n.label = k
            values.append(n)
        return Node(sorted(values, key=lambda node: node.label))

    elif isinstance(value, list) and len(value) > 0:
        return ListNode([nodefy(v) for v in value])

    elif isinstance(value, Node):
        return value

    return Node(value)
