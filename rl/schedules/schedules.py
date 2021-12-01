from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay as KerasPolynomialDecay

@tf.keras.utils.register_keras_serializable()
class PolynomialDecay(KerasPolynomialDecay):
	def __init__(self, step_counter:tf.Tensor, **kwargs) -> None:
		super(PolynomialDecay, self).__init__(**kwargs)
		self.step_counter = step_counter
	def __call__(self, _):
		return super(PolynomialDecay, self).__call__(self.step_counter)
	def get_config(self) -> Dict[str, Any]:
		config = super(PolynomialDecay, self).get_config()
		config["step_counter"] = self.step_counter
		return config