import tensorflow as tf


class Encoder(tf.Module):
    def __init__(
        self, vocab_size, embedding_dim, enc_units, batch_size, name="encoder"
    ):
        super(Encoder, self).__init__(name=name)
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, name="embedding"
        )
        self.conv1d = []
        self.batch_norm = []
        for i in range(1, 4):
            self.conv1d.append(
                tf.keras.layers.Conv1D(
                    filters=enc_units,
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
                units=enc_units,
                return_sequences=True,
                return_state=True,
                name="bi_lstm",
            ),
            name="bi_lstm",
        )

    def __call__(self, inputs, training=False):
        x = self.embedding(inputs)
        for i in range(3):
            x = self.conv1d[i](x)
            x = self.batch_norm[i](x, training=training)
        x = self.bi_lstm(x)
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
        # self.attension = LuongAttension(units=dec_units, name="attension")
        self.attension = tf.keras.layers.Attention(use_scale=True, name="attension")
        self.decoder_rnn = tf.keras.layers.GRU(
            units=dec_units,
            return_sequences=True,
            return_state=True,
            name="decoder_rnn",
        )
        self.fc = tf.keras.layers.Dense(
            units=reduction, activation=tf.nn.relu, name="fc"
        )

    def __call__(self, enc_output, dec, training=False):
        x = self.prenet(dec, training=training)
        x, state = self.attension_rnn(x, initial_state=enc_output)
        context_vector, alignment = self.attension([x

class Tacotron2(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        enc_units,
        dec_units,
        reduction,
        batch_size,
        name="tacotron2",
    ):
        super(Tacotron2, self).__init__(name=name)
        self.batch_size = batch_size
        self.reduction = reduction
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            enc_units=enc_units,
            batch_size=batch_size,
            name="encoder",
        )
        self.decoder = Decoder(
            dec_units=dec_units,
            reduction=reduction,
            batch_size=batch_size,
            name="decoder",
        )
        self.fc = tf.keras.layers.Dense(
            units=vocab_size, activation=tf.nn.softmax, name="fc"
        )

    def __call__(self, text, dec, training=False):
        enc_output = self.encoder(text, training=training)
        dec_output, alignment = self.decoder(enc_output, dec, training=training)
        output = self.fc(dec_output)
        return output, alignment
