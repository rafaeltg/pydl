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
    def _obj_fn(x, hp_space, data_x, data_y, *args):
        pass


class CVObjectiveFunction(ObjectiveFunction):

    def __init__(self, scoring=None, cv_method='split', **kwargs):
        super().__init__()
        self._args += tuple([CV(method=cv_method, **kwargs), scoring])

    @staticmethod
    def _obj_fn(x, hp_space, data_x, data_y, *args):
        cv = args[0]
        scoring = args[1]
        m = load_model(hp_space.get_value(x))
        res = cv.run(model=m, x=data_x, y=data_y, scoring=scoring, max_thread=1)
        s = cv.get_scorer_name(scoring) if scoring is not None else m.get_loss_func()
        return res[s]['mean']
