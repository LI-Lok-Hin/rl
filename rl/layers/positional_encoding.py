import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class PositionalEncoding(Layer):
    def build(self, input_shape) -> None:
        input_len = input_shape[1]
        self.pos = tf.one_hot(
            indices=tf.range(input_len),
            depth=input_len
        )
        self.pos = tf.cast(self.pos, tf.float32)
        self.pos = tf.expand_dims(self.pos, axis=0)
        
    def call(self, inputs):
        batch_size = K.shape(inputs)[0]
        pos = tf.repeat(self.pos, repeats=[batch_size], axis=0)
        outputs = tf.concat([inputs, pos], axis=-1)
        return outputs
