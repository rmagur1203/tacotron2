import numpy as np
import tensorflow as tf


class PreNet(tf.Module):
    def __init__(self, out_units, dropout_rate=0.5, name="prenet"):
        super(PreNet, self).__init__(name=name)
        self.fc1 = tf.keras.layers.Dense(
            units=out_units[0], activation=tf.nn.relu, name="fc1"
        )
        self.fc2 = tf.keras.layers.Dense(
            units=out_units[1], activation=tf.nn.relu, name="fc2"
        )
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name="dropout")

    def __call__(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units, name="luong_attention"):
        super(LuongAttention, self).__init__(name=name)
        self.W = tf.keras.layers.Dense(units=units, name="W")

    def call(self, query, value):
        alignment = tf.nn.softmax(tf.matmul(query, self.W(value), transpose_b=True))
        context = tf.matmul(alignment, value)
        context = tf.concat([context, query], axis=-1)
        alignment = tf.transpose(alignment, [0, 2, 1])
        return context, alignment


class LocationLayer(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernal_size, attention_dim, name="location_layer"):
        super(LocationLayer, self).__init__(name=name)
        self.conv = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=kernal_size,
            padding="same",
            use_bias=False,
            strides=1,
            dilation_rate=1,
            name="location_conv",
        )
        self.dense = tf.keras.layers.Dense(
            units=attention_dim,
            activation=tf.nn.tanh,
            use_bias=False,
            name="location_dense",
        )

    def call(self, inputs):
        x = self.conv(inputs)
        x = np.transpose(x, [0, 2, 1])
        x = self.dense(x)
        return x


class LocationSensitiveAttention(tf.keras.layers.Layer):
    def __init__(
        self, attention_dim, filters, kernal_size, name="location_sensitive_attention"
    ):
        super(LocationSensitiveAttention, self).__init__(name=name)
        self.query_layer = tf.keras.layers.Dense(
            units=attention_dim,
            use_bias=False,
            name="query_layer",
            activation=tf.nn.tanh,
        )
        self.memory_layer = tf.keras.layers.Dense(
            units=attention_dim,
            use_bias=False,
            name="memory_layer",
            activation=tf.nn.tanh,
        )
        self.V = tf.keras.layers.Dense(
            units=1, use_bias=False, name="V"
        )  # score = V * tanh(query + memory)
        self.location_layer = LocationLayer(
            n_filters=filters,
            kernal_size=kernal_size,
            attention_dim=attention_dim,
            name="location",
        )

    def call(self, query, value, attention_weights_cat):
        query = self.query_layer(query)
        value = self.memory_layer(value)
        location = self.location_layer(attention_weights_cat)
        score = self.V(tf.nn.tanh(query + value + location))
        alignment = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(alignment * value, axis=1)
        alignment = tf.transpose(alignment, [0, 2, 1])
        return context, alignment
