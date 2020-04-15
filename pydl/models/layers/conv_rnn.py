import keras.layers as kl


class ConvLSTM1D(kl.ConvLSTM2D):

    def __init__(self,
                 filters,
                 kernel_size,
                 **kwargs):

        super().__init__(
            filters=filters,
            kernel_size=kernel_size if isinstance(kernel_size, tuple) or isinstance(kernel_size, list) else (1, kernel_size),
            **kwargs
        )
