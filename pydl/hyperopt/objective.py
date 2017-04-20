from ..models.utils import load_model
from ..model_selection import CV


class ObjectiveFunction:

    def __init__(self):
        self._args = ()

    @property
    def args(self):
        return self._args

    @property
    def obj_fn(self):
        return self._obj_fn

    @staticmethod
    def _obj_fn(x):
        pass


class CVObjectiveFunction(ObjectiveFunction):

    def __init__(self, scoring=None, cv_method='split', **kwargs):
        super().__init__()
        self._args += tuple([CV(method=cv_method, **kwargs), scoring])

    @staticmethod
    def child_initialize(_hp_space, _x, _y, _cv, _scoring):
        global hp_space, data_x, data_y, cv, scoring
        hp_space = _hp_space
        data_x = _x
        data_y = _y
        cv = _cv
        scoring = _scoring

    @staticmethod
    def _obj_fn(x):
        m = load_model(hp_space.get_value(x))
        res = cv.run(model=m, x=data_x, y=data_y, scoring=scoring, max_threads=1)
        s = cv.get_scorer_name(scoring) if scoring is not None else m.get_loss_func()
        return res[s]['mean']
