import tensorflow as tf

from models.module import LocationSensitiveAttention, PreNet


class Encoder(tf.Module):
    def __init__(
        self,
        vocab_size,
        batch_size,
        embedding_dim=512,
        lstm_units=512,
        conv_channels=512,
        conv_layers=3,
        name="encoder",
    ):
        super(Encoder, self).__init__(name=name)
        self.batch_size = batch_size
        self.conv_layers = conv_layers
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, name="embedding"
        )
        self.conv1d = []
        self.batch_norm = []
        for i in range(1, conv_layers + 1):
            self.conv1d.append(
                tf.keras.layers.Conv1D(
                    filters=conv_channels,
                    kernel_size=5,
                    strides=1,
                    padding="same",
                    name="conv1d_{}".format(i),
                )
            )
            self.batch_norm.append(
                tf.keras.layers.BatchNormalization(name="batch_norm_{}".format(i))
            )
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=lstm_units,
                return_sequences=True,
                return_state=True,
                name="bi_lstm",
            ),
            name="bi_lstm",
        )

    def __call__(self, inputs, training=False):
        x = self.embedding(inputs)
        for i in range(self.conv_layers):
            x = self.conv1d[i](x)
            x = self.batch_norm[i](x, training=training)
        x = self.bi_lstm(x)
        return x


class Decoder(tf.keras.Model):
    def __init__(
        self,
        batch_size,
        dec_units,
        reduction,
        attention_dim,
        mel_dim=80,
        name="decoder",
    ):
        super(Decoder, self).__init__(name=name)
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.prenet = PreNet([256, 256], name="prenet")
        self.attension_rnn = tf.keras.layers.LSTM(
            units=attention_dim,
            return_sequences=True,
            return_state=True,
            name="attension_rnn",
        )
        self.attention = LocationSensitiveAttention(
            attention_dim=attention_dim, filters=32, kernel_size=31, name="attention"
        )
        self.decoder_rnn = tf.keras.layers.LSTM(
            units=dec_units,
            return_sequences=True,
            return_state=True,
            name="decoder_rnn",
        )
        self.projection = tf.keras.layers.Dense(
            units=mel_dim * reduction, activation=tf.nn.relu, name="projection"
        )
        self.stop_token = tf.keras.layers.Dense(
            units=1, activation=tf.nn.sigmoid, name="stop_token"
        )

    def __call__(self, enc_output, dec, training=False):
        x = self.prenet(dec, training=training)
        x, state_h, state_c = self.attension_rnn(x, training=training)
        context, alignment = self.attention(x, enc_output)
        x = tf.concat([context, x], axis=-1)
        x, state_h, state_c = self.decoder_rnn(x, training=training)
        x = self.projection(x)
        stop_token = self.stop_token(x)
        return x, stop_token, alignment


class Tacotron2(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        reduction,
        batch_size,
        name="tacotron2",
    ):
        super(Tacotron2, self).__init__(name=name)
        self.batch_size = batch_size
        self.reduction = reduction
        self.encoder = Encoder(
            vocab_size=vocab_size,
            batch_size=batch_size,
            name="encoder",
        )
        self.decoder = Decoder(
            dec_units=1024,
            reduction=reduction,
            batch_size=batch_size,
            attention_dim=128,
            name="decoder",
        )

    def __call__(self, text, dec, training=False):
        enc_output = self.encoder(text, training=training)
        mel, stop_token, alignment = self.decoder(enc_output, dec, training=training)
        return mel, stop_token, alignment
