from typing import Iterable, Union

from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed

class CNNBlock(Layer):
	def __init__(
		self,
		conv_shapes,
		fc_sizes: Iterable[int],
		is_lstm: bool = False,
		return_sequences: bool = False,
		activation: Union[str, Activation] = "relu",
		kernel_initializer: Union[str, Initializer] = "glorot_uniform",
		bias_initializer: Union[str, Initializer] = "zeros",
		**kwargs
	) -> None:
		super(CNNBlock, self).__init__(**kwargs)
		self.conv_shapes = conv_shapes
		self.fc_sizes = fc_sizes
		self.is_lstm = is_lstm
		self.return_sequences = return_sequences
		self.activation = activation
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.layers = []
		for conv_shape in conv_shapes:
			filters, kernel_size, strides = conv_shape
			layer = Conv2D(
				filters=filters,
				kernel_size=kernel_size,
				strides=strides,
				activation=activation,
				kernel_initializer=kernel_initializer,
				bias_initializer=bias_initializer
			)
			if is_lstm:
				layer = TimeDistributed(layer)
			self.layers.append(layer)
		
		layer = Flatten()
		if is_lstm:
			layer = TimeDistributed(layer)
		self.layers.append(layer)
		
		for i, fc_size in enumerate(fc_sizes):
			if is_lstm:
				if i == len(fc_sizes) - 1:
					return_sequences = self.return_sequences
				else:
					return_sequences = True
				layer = LSTM(
					fc_size,
					kernel_initializer=kernel_initializer,
					bias_initializer=bias_initializer,
					return_sequences=return_sequences
				)
			else:
				layer = Dense(
					fc_size,
					activation=activation,
					kernel_initializer=kernel_initializer,
					bias_initializer=bias_initializer
				)
			self.layers.append(layer)

	def call(self, inputs):
		outputs = inputs
		for layer in self.layers:
			outputs = layer(outputs)
		return outputs