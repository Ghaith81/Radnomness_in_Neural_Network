import tensorflow as tf


# Define a dropconnect layer
class DropConnect(tf.keras.layers.Wrapper):
    def __init__(self, layer, prob, **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def call(self, x, training=None):
        if training and self.prob < 1:
            keep_prob = 1 - self.prob
            mask = tf.random.uniform(shape=tf.shape(x)) < keep_prob
            mask = tf.cast(mask, dtype=tf.float32)
            #tf.print('before', x)
            x = x * mask
            #tf.print('after', x)
            return self.layer(x)
        else:
            return tf.identity(self.layer(x))


class NoisyReLU(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(NoisyReLU, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            #noise = tf.random.normal(shape=tf.shape(inputs), mean=0., stddev=self.stddev)
            #tf.print("noise added", tf.shape(inputs))
            return tf.nn.relu(inputs + tf.random.normal(shape=tf.shape(inputs), mean=0., stddev=self.stddev))
        else:
            #print("no noise")
            return tf.nn.relu(inputs)