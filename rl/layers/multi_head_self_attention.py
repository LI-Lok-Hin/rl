from typing import Iterable, Union

from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.regularizers import Regularizer

class MultiHeadSelfAttention(Layer):
	def __init__(
		self,
		num_heads: int,
		value_dim: int = None,
		dropout: float = 0.0,
		use_bias: bool = True,
		output_shape = None,
		attention_axes: Iterable[int] = None,
		kernel_initializer: Union[str, Initializer] = "glorot_uniform",
		bias_initializer: Union[str, Initializer] = "zeros",
		kernel_regularizer: Union[str, Regularizer] = None,
		bias_regularizer: Union[str, Regularizer] = None,
		activity_regularizer: Union[str, Regularizer] = None,
		kernel_constraint: Union[str, Constraint] = None,
		bias_constraint: Union[str, Constraint] = None,
		**kwargs
	) -> None:
		super().__init__(**kwargs)
		self.num_heads = num_heads
		self.value_dim = value_dim
		self.dropout = dropout
		self.use_bias = use_bias
		self.attention_output_shape = output_shape
		self.attention_axes = attention_axes
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.kernel_regularizer = kernel_regularizer
		self.bias_regularizer = bias_regularizer
		self.activity_regularizer = activity_regularizer
		self.kernel_constraint = kernel_constraint
		self.bias_constraint = bias_constraint

	def build(self, input_shape) -> None:
		key_dim = input_shape[-1]
		self.attention_layer = MultiHeadAttention(
			num_heads=self.num_heads,
			key_dim=key_dim,
			value_dim=self.value_dim,
			dropout=self.dropout,
			use_bias=self.use_bias,
			output_shape=self.attention_output_shape,
			attention_axes=self.attention_axes,
			kernel_initializer=self.kernel_initializer,
			bias_initializer=self.bias_initializer,
			kernel_regularizer=self.kernel_regularizer,
			bias_regularizer=self.bias_regularizer,
			activity_regularizer=self.activity_regularizer,
			kernel_constraint=self.kernel_constraint,
			bias_constraint=self.bias_constraint
		)

	def call(
		self,
		inputs,
		attention_mask = None,
		return_attention_scores: bool = False,
		training: bool = None
	):
		outputs = self.attention_layer(
			query=inputs,
			value=inputs,
			attention_mask=attention_mask,
			return_attention_scores=return_attention_scores,
			training=training
		)
		return outputs