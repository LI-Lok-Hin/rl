import tensorflow as tf
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Subtract

class DuelingLayer(Layer):
	def __init__(self, output_shape, **kwargs):
		super(DuelingLayer, self).__init__(**kwargs)
		self.value = Dense(1, activation="linear")
		self.advantage = Dense(output_shape[0], activation="linear")
		self.subtract = Subtract()
		self.add = Add()
        
	def call(self, inputs):
		value = self.value(inputs)
		advantage = self.advantage(inputs)
		average = tf.reduce_mean(advantage, axis=1, keepdims=True)
		subtract = self.subtract([advantage, average])
		add = self.add([value, subtract])
		return add