import tensorflow as tf
from tensorflow import keras


class Conv1D_Bank(keras.Model):
    def __init__(self, K, filters=128):
        super(Conv1D_Bank, self).__init__()
        self.K = K
        self.conv1d_layers = [keras.layers.Conv1D(filters=filters, kernel_size=k, padding='same')
                              for k in range(1, K+1)]
        self.batch_norm_layers = [keras.layers.BatchNormalization()
                                  for _ in range(K)]

    def call(self, inputs, training=None):
        x = tf.concat([conv1d(inputs)
                      for conv1d in self.conv1d_layers], axis=-1)
        x = tf.concat([batch_norm(x, training=training)
                       for batch_norm in self.batch_norm_layers], axis=-1)
        x = tf.nn.relu(x)
        return x


class Conv1D_Projection(keras.Model):
    def __init__(self, projections, kernel_size=3):
        super(Conv1D_Projection, self).__init__()
        self.conv1d_1 = keras.layers.Conv1D(
            filters=projections[0], kernel_size=kernel_size, padding='same')
        self.batch_norm_1 = keras.layers.BatchNormalization()
        self.conv1d_2 = keras.layers.Conv1D(
            filters=projections[1], kernel_size=kernel_size, padding='same')
        self.batch_norm_2 = keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        x = self.conv1d_1(inputs)
        x = self.batch_norm_1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv1d_2(x)
        x = self.batch_norm_2(x, training=training)
        return x


class HighwayNet(keras.Model):
    '''
    Highway Networks
    Args:
        units: number of units in the dense layer

    Input shape: (batch_size, time_steps, input_dim)
    Output shape: (batch_size, time_steps, output_dim)

    Reference:
        https://arxiv.org/pdf/1505.00387.pdf
    '''

    def __init__(self, units):
        super(HighwayNet, self).__init__()
        self.units = units
        self.H = keras.layers.Dense(units=units, activation='relu')
        self.T = keras.layers.Dense(units=units, activation='sigmoid')

    def call(self, inputs, training=None):
        x = inputs
        t = self.T(x)
        h = self.H(x)
        x = t * h + (1 - t) * x
        return x


class CBHG(keras.Model):
    '''
    CBHG: Convolutional Bank, Highway Networks, and GRU
    Args:
        K: number of convolutional bank
        projections: list of number of filters for convolutional projections

    Input shape: (batch_size, time_steps, input_dim)
    Output shape: (batch_size, time_steps, output_dim)

    Reference:
        https://arxiv.org/pdf/1703.10135.pdf
    '''

    def __init__(self, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.K = K
        self.projections = projections
        self.conv_bank = Conv1D_Bank(K)
        self.max_pool = keras.layers.MaxPool1D(
            pool_size=2, strides=1, padding='same')

        # Convolutional projections
        self.conv_projections = Conv1D_Projection(
            projections)

        # Highway Networks
        self.highwaynet = HighwayNet(projections[1])

        # Bidirectional GRU
        self.gru = keras.layers.Bidirectional(
            keras.layers.GRU(projections[1], return_sequences=True)
        )

    def call(self, inputs, training=None):
        x = self.conv_bank(inputs)
        x = self.max_pool(x)
        x = self.conv_projections(x)
        x = x + inputs
        x = self.highwaynet(x)
        x, _ = self.gru(x)
        return x
