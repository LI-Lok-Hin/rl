import tensorflow as tf
from tensorflow.keras.layers import Layer

class SqueezeLayer(Layer):
	def call(self, inputs):
		return tf.squeeze(inputs)