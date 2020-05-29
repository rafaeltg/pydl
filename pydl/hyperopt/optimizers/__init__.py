from .optimizer import Optimizer
from .cmaes import CMAES


__all__ = [
    'Optimizer',
    'CMAES',
    'optimizer_from_config'
]


_objs = {
    CMAES.__name__: CMAES
}


def optimizer_from_config(config: dict) -> [Optimizer]:
    if len(config) == 0:
        return None

    cls_name = config.get("class_name", None)
    if cls_name is None:
        return None

    cls = _objs.get(cls_name, None)
    if cls is None:
        return None

    cfg = config.get("config", dict())

    if len(cfg) > 0 and hasattr(cls, "from_config"):
        return cls.from_config(cfg)

    return cls(**cfg)
