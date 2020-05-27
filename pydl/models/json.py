import os
from json import JSONEncoder, dump


__all__ = [
    'ModelJsonEncoder',
    'save_json'
]


class ModelJsonEncoder(JSONEncoder):

    def default(self, o):
        if hasattr(o, 'get_config'):
            return {
                'class_name': o.__class__.__name__,
                'config': o.get_config()
            }

        return super().default(o)


def save_json(data, file_path, sort_keys=True, indent=2, ensure_ascii=False, cls=ModelJsonEncoder):
    directory = os.path.dirname(file_path)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w', encoding='utf-8') as outfile:
        dump(data,
             outfile,
             sort_keys=sort_keys,
             indent=indent,
             ensure_ascii=ensure_ascii,
             separators=(',', ': '),
             cls=cls)
