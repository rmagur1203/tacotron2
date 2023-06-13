import tensorflow as tf
from models.cbhg import CBHG

from models.module import LuongAttension, PreNet


# tacotron Encoder
class Encoder(tf.keras.Module):
    def __init__(
        self, vocab_size, embedding_dim, enc_units, batch_size, name="encoder"
    ):
        super(Encoder, self).__init__(name=name)
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, name="embedding"
        )
        self.prenet = PreNet([256, 128], name="prenet")
        self.cbhg = CBHG(
            out_units=[enc_units, enc_units],
            conv_channels=[128, enc_units],
            name="cbhg",
        )

    def __call__(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.prenet(x, training=training)
        x = self.cbhg(x, training=training)
        return x


class Decoder(tf.keras.Model):
    def __init__(
        self, vocab_size, embedding_dim, dec_units, batch_size, name="decoder"
    ):
        super(Decoder, self).__init__(name=name)
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, name="embedding"
        )
        self.prenet = PreNet([256, 128], name="prenet")
        self.attension = LuongAttension(units=dec_units, name="attension")
        # self.attension = tf.keras.layers.Attention(use_scale=True, name='attension')
        self.gru = tf.keras.layers.GRU(
            units=dec_units, return_sequences=True, return_state=True, name="gru"
        )
        self.fc = tf.keras.layers.Dense(units=vocab_size, name="fc")

    def __call__(self, inputs, enc_output, training=False):
        x = self.embedding(inputs)
        x = self.prenet(x, training=training)
        context_vector, alignment = self.attension(x, enc_output)
        x, state = self.gru(context_vector)
        x = self.fc(x)
        return x, state, alignment
