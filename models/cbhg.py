import tensorflow as tf


class CBHG(tf.Module):
    def __init__(self, out_units, conv_channels, name="cbhg"):
        super(CBHG, self).__init__(name=name)
        self.conv1d_bank = Conv1DBank(
            K=16, conv_channels=conv_channels[0], name="conv1d_bank"
        )
        self.max_pool1d = tf.keras.layers.MaxPool1D(
            pool_size=2, strides=1, padding="same", name="max_pool1d"
        )
        self.conv1d_projections = Conv1DProjections(
            conv_channels=conv_channels[1], name="conv1d_projections"
        )
        self.highwaynet = HighwayNet(
            out_units=out_units[0], num_units=out_units[1], name="highwaynet"
        )
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=out_units[1], return_sequences=True),
            name="bi_gru",
        )

    def __call__(self, inputs, training=False):
        x = self.conv1d_bank(inputs)
        x = self.max_pool1d(x)
        x = self.conv1d_projections(x)
        x = x + inputs
        x = self.highwaynet(x)
        x = self.bi_gru(x)
        return x


# activation: relu
class Conv1DBank(tf.keras.layers.Layer):
    def __init__(self, K, conv_channels, activation=tf.nn.relu, name="conv1d_bank"):
        super(Conv1DBank, self).__init__(name=name)
        self.K = K
        self.conv_channels = conv_channels
        self.conv1ds = [
            tf.keras.layers.Conv1D(
                filters=conv_channels,
                kernel_size=k,
                activation=activation,
                padding="same",
                name="conv1d_{}".format(k),
            )
            for k in range(1, K + 1)
        ]

    def __call__(self, inputs):
        outputs = [conv1d(inputs) for conv1d in self.conv1ds]
        outputs = tf.concat(outputs, axis=-1)
        return outputs


class Conv1DProjections(tf.keras.layers.Layer):
    def __init__(self, conv_channels, name="conv1d_projections"):
        super(Conv1DProjections, self).__init__(name=name)
        self.conv1d_1 = tf.keras.layers.Conv1D(
            filters=conv_channels, kernel_size=3, padding="same", name="conv1d_1"
        )
        self.conv1d_2 = tf.keras.layers.Conv1D(
            filters=conv_channels, kernel_size=3, padding="same", name="conv1d_2"
        )

    def __call__(self, inputs):
        x = self.conv1d_1(inputs)
        x = tf.nn.relu(x)
        x = self.conv1d_2(x)
        return x


class HighwayNet(tf.keras.layers.Layer):
    def __init__(self, out_units, num_units, name="highwaynet"):
        super(HighwayNet, self).__init__(name=name)
        self.H = [
            tf.keras.layers.Dense(
                units=num_units, activation=tf.nn.relu, name="H_{}".format(i)
            )
            for i in range(4)
        ]
        self.T = [
            tf.keras.layers.Dense(
                units=num_units, activation=tf.nn.sigmoid, name="T_{}".format(i)
            )
            for i in range(4)
        ]

    def __call__(self, inputs):
        x = inputs
        for i in range(4):
            h = self.H[i](x)
            t = self.T[i](x)
            x = h * t + x * (1.0 - t)
        return x
