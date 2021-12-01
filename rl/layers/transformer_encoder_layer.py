from typing import Iterable, Union

from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.regularizers import Regularizer
from rl.layers import MultiHeadSelfAttention

class TransformerEncoderLayer(Layer):
	def __init__(
		self,
		n_hidden: int,
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
		self.kernel_regularizer = kernel_regularizer
		self.attention_layer = MultiHeadSelfAttention(
			num_heads=num_heads,
			value_dim=value_dim,
			dropout=dropout,
			use_bias=use_bias,
			output_shape=output_shape,
			attention_axes=attention_axes,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			kernel_constraint=kernel_constraint,
			bias_constraint=bias_constraint
		)
		self.attention_layer_norm = LayerNormalization(epsilon=1e-8)
		self.hidden_fc = Dense(n_hidden, activation="elu", kernel_regularizer=kernel_regularizer)
		self.fc_layer_norm = LayerNormalization(epsilon=1e-8)
		
	def build(self, input_shape) -> None:
		feature_size = input_shape[-1]
		self.fc = Dense(feature_size, kernel_regularizer=self.kernel_regularizer)

	def call(
		self,
		inputs,
		attention_mask = None,
		return_attention_scores = False,
		training = None
	):
		attention_outputs = self.attention_layer(
			inputs=inputs,
			attention_mask=attention_mask,
			return_attention_scores=return_attention_scores,
			training=training
		)
		norm_attention_outputs = self.attention_layer_norm(
			inputs + attention_outputs
		)

		hidden_outputs = self.hidden_fc(norm_attention_outputs)
		fc_outputs = self.fc(hidden_outputs)
		outputs = self.fc_layer_norm(
			norm_attention_outputs + fc_outputs
		)
		return outputs