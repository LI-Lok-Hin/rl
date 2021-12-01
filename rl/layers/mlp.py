from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Dense, Layer
from typing import Iterable, Union


class MLP(Layer):
    def __init__(
        self,
        fc_sizes: Iterable[int],
        activation: str = "relu",
		kernel_initializer: Union[str, Initializer] = "glorot_uniform",
		bias_initializer: Union[str, Initializer] = "zeros",
		**kwargs
    ) -> None:
        super(MLP, self).__init__(**kwargs)
        self.fc_sizes = fc_sizes
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.layers = []
        for fc_size in fc_sizes:
            self.layers.append(
                Dense(
                    units=fc_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer
                )
            )
    
    def call(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs