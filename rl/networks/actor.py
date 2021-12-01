from typing import Callable, List, Union

import gym
from gym.spaces import Discrete
import numpy as np
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Input, Dense, Flatten, Layer

from rl.networks import Network

class Actor(Network):
	def __init__(
		self,
		input_shape,
		output_space: gym.Space,
		layers: List[Layer],
		activation: Union[Callable, str] = None,
		kernel_initializer: Union[Initializer, str] = "glorot_uniform",
		bias_initializer: Union[Initializer, str] = "zeros",
		name: str = None
	) -> None:
		if isinstance(output_space, Discrete):
			output_shape = output_space.n
		else:
			output_shape = np.prod(output_space.shape)
		inputs = Input(shape=input_shape)
		l = inputs
		for layer in layers:
			l = layer(l)
		l = Flatten()(l)
		outputs = Dense(
			output_shape,
			activation=activation,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer
		)(l)
		super().__init__(
			inputs=inputs,
			outputs=outputs,
			name=name
		)