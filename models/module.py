import tensorflow as tf

class PreNet(tf.Module):
    def __init__(self, out_units, dropout_rate=0.5, name='prenet'):
        super(PreNet, self).__init__(name=name)
        self.fc1 = tf.keras.layers.Dense(units=out_units[0], activation=tf.nn.relu, name='fc1')
        self.fc2 = tf.keras.layers.Dense(units=out_units[1], activation=tf.nn.relu, name='fc2')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
    
    def __call__(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x

class LuongAttension(tf.keras.layers.Layer):
    def __init__(self, units, name='luong_attension'):
        super(LuongAttension, self).__init__(name=name)
        self.W = tf.keras.layers.Dense(units=units, name='W')

    def __call__(self, query, values):
        score = tf.matmul(query, self.W(values), transpose_b=True)
        alignment = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(alignment, values)
        return context_vector, alignment