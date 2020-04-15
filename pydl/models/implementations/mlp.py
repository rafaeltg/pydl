from ..base import SupervisedModel
from ..layers import Dense, Dropout


class MLP(SupervisedModel):

    """
        Multi-Layer Perceptron
    """

    @classmethod
    def is_valid_layer(cls, layer):
        return isinstance(layer, Dense) or isinstance(layer, Dropout)
