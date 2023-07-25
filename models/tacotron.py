import tensorflow as tf
from models.cbhg import CBHG

from models.module import LuongAttention, PreNet


# tacotron Encoder
class Encoder(tf.Module):
    def __init__(
        self, vocab_size, embedding_dim, enc_units, batch_size, name="encoder"
    ):
        super(Encoder, self).__init__(name=name)
        self.batch_size = batch_size
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
        self,
        dec_units,
        reduction,
        batch_size,
        name="decoder",
    ):
        super(Decoder, self).__init__(name=name)
        self.batch_size = batch_size
        self.dec_units = dec_units
        # self.embedding = tf.keras.layers.Embedding(
        #     vocab_size, embedding_dim, name="embedding"
        # )
        self.prenet = PreNet([256, 128], name="prenet")
        self.attension_rnn = tf.keras.layers.GRU(
            units=dec_units,
            return_sequences=True,
            return_state=True,
            name="attension_rnn",
        )
        self.attension = LuongAttention(units=dec_units, name="attention")
        # self.attension = tf.keras.layers.Attention(use_scale=True, name="attension")
        self.decoder_rnn = tf.keras.layers.GRU(
            units=dec_units,
            return_sequences=True,
            return_state=True,
            name="decoder_rnn",
        )
        self.fc = tf.keras.layers.Dense(units=80 * reduction, name="fc")

    def __call__(self, inputs, enc_output, training=False):
        # x = self.embedding(inputs)
        x = self.prenet(inputs, training=training)
        x, state = self.attension_rnn(x)
        context_vector, alignment = self.attension(x, enc_output)
        x, state = self.decoder_rnn(context_vector)
        x = self.fc(x)
        x = tf.reshape(x, (self.batch_size, -1, 80))
        return x, state, alignment


class Tacotron(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        enc_units,
        dec_units,
        batch_size,
        reduction=5,
        name="tacotron",
    ):
        super(Tacotron, self).__init__(name=name)
        self.batch_size = batch_size
        self.encoder = Encoder(
            vocab_size, embedding_dim, enc_units, batch_size, name="encoder"
        )
        self.decoder = Decoder(dec_units, reduction, batch_size, name="decoder")

    def __call__(self, enc_inputs, dec_inputs, training=False):
        enc_output = self.encoder(enc_inputs, training=training)
        dec_output, state, alignment = self.decoder(
            dec_inputs, enc_output, training=training
        )
        return dec_output, alignment


class PostNet(tf.keras.Model):
    def __init__(self, mel_dim, n_fft, conv_dim=[256, 80], name="postnet"):
        super(PostNet, self).__init__(name=name)
        self.cbhg = CBHG(
            out_units=[mel_dim, mel_dim], conv_channels=conv_dim, name="cbhg"
        )
        self.fc = tf.keras.layers.Dense(units=n_fft // 2 + 1, name="fc")

    def __call__(self, inputs, training=False):
        x = self.cbhg(inputs, training=training)
        x = self.fc(x)
        return x
