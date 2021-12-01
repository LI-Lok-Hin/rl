from typing import Iterable, Union

from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Input, Dense, Flatten, Layer

from rl.layers import SqueezeLayer
from rl.networks.network import Network

class Critic(Network):
	def __init__(
		self,
		input_shape,
		layers: Iterable[Layer],
		kernel_initializer: Union[Initializer, str] = "glorot_uniform",
		bias_initializer: Union[Initializer, str] = "zeros",
		name: str = None
	) -> None:
		inputs = Input(shape=input_shape)
		l = inputs
		for layer in layers:
			l = layer(l)
		l = Flatten()(l)
		l = Dense(
			units=1,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer
		)(l)
		outputs = SqueezeLayer()(l)
		super().__init__(
			inputs=inputs,
			outputs=outputs,
			name=name
		)