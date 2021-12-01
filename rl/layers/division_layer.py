from tensorflow.keras.layers import Layer

class DivisionLayer(Layer):
	def __init__(
        self,
        divisor:float,
        **kwargs
    ) -> None:
		super(DivisionLayer, self).__init__(**kwargs)
		self.divisor = divisor
		
	def call(self, inputs):
		return inputs / self.divisor