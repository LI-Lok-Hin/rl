import os
import pickle
import tempfile
from typing import Any, Dict, Union

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss

class Network(Model):
	def __init__(
		self,
		inputs,
		outputs,
		optimizer: Union[str, Optimizer] = None,
		loss: Union[str, Loss] = None,
		name: str = None
	) -> None:
		super(Network, self).__init__(
			inputs=inputs,
			outputs=outputs,
			name=name
		)
		self.optimizer = optimizer
		self.loss = loss
	
	@tf.function
	def call(
		self,
		inputs,
		training = None
	):
		outputs = inputs
		for layer in self.layers:
			if layer.trainable:
				outputs = layer(
					outputs,
					training = training
				)
			else:
				outputs = layer(outputs)
		return outputs

	def __getstate__(self) -> Dict[str, Any]:
		return self.get_config()

	def __setstate__(self, state:Dict[str, Any]) -> None:
		obj = Network.from_config(state)
		self.__dict__.update(obj.__dict__)

	def get_config(self) -> Dict[str, Any]:
		config = {
			"inputs" : self.inputs,
			"outputs" : self.outputs,
			"optimizer" : self.optimizer,
			"loss" : self.loss,
			"name" : self.name,
			"class" : self.__class__
		}
		return config

	@classmethod
	def from_config(cls, config:Dict[str, Any]) -> "Network":
		class_ = config.pop("class")
		network = cls(**config)
		network.__class__ = class_
		return network

	def clone_config(self) -> "Network":
		with tempfile.TemporaryFile() as f:
			pickle.dump(
				obj = self.get_config(),
				file = f,
				protocol = pickle.HIGHEST_PROTOCOL
			)
			f.seek(0)
			config = pickle.load(f)
		return Network.from_config(config)

	def clone_weight(self, other:"Network") -> None:
		self.set_weights(other.get_weights())
		
	def save_config(self, filepath:str) -> None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		with open(filepath + ".pkl", "wb") as f:
			pickle.dump(
				obj = self.get_config(),
				file = f,
				protocol = pickle.HIGHEST_PROTOCOL
			)

	@classmethod
	def load_config(cls, filepath:str) -> "Network":
		with open(filepath + ".pkl", "rb") as f:
			config = pickle.load(f)
		return cls.from_config(config)
