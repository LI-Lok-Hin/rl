from typing import Iterable, Union
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from rl.layers import DuelingLayer
from rl.networks.network import Network

class RNN(Network):
	def __init__(
		self,
		input_shape,
		output_shape,
		preprocessing_layers: Iterable[layers.Layer] = [],
		conv_shapes = [],
		fc_sizes: Iterable[int] = [],
		is_deuling: bool = True,
		optimizer: Union[str, Optimizer] = "adam",
		loss: Union[str, Loss] = None,
		name: str = None,
	) -> None:
		inputs = layers.Input(shape=input_shape)
		l = inputs

		for pre_layer in preprocessing_layers:
			l = pre_layer(l)

		for conv_shape in conv_shapes:
			filters, kernel_size, strides = conv_shape
			l = layers.TimeDistributed(layers.Conv2D(
				filters=filters,
				kernel_size=kernel_size,
				strides=strides,
				activation="relu"
			))(l)

		l = layers.TimeDistributed(layers.Flatten())(l)

		for fc_size in fc_sizes:
			l = layers.LSTM(fc_size)(l)

		if is_deuling:
			outputs = DuelingLayer(output_shape)(l)
		else:
			outputs = layers.Dense(output_shape[0], activation="linear")(l)
		super().__init__(
			inputs=inputs,
			outputs=outputs,
			optimizer=optimizer,
			loss=loss,
			name=name
		)

	@tf.function
	def train(self, x, target_y, masks) -> None:
		with tf.GradientTape() as tape:
			pred_y = self(x, training=True)
			pred_y = tf.reduce_sum(tf.multiply(pred_y, masks), axis=1)
			loss = self.loss(target_y, pred_y)
		grads = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
	