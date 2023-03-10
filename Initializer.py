import tensorflow as tf

class FixedWeightInitializer(tf.keras.initializers.Initializer):
    def __init__(self, percentage=0.0, std=0.1):
        self.percentage = percentage
        self.std = std

    def __call__(self, shape, dtype=None):
        total_weights = 1
        for dim in shape:
            #tf.print(dim)
            total_weights *= dim
        #tf.print(total_weights)
        not_trainable_weights = int(total_weights * self.percentage)
        trainable_weights = total_weights - not_trainable_weights
        not_trainable_weights = tf.random.normal(shape=(not_trainable_weights,), mean=0, stddev=self.std, dtype=dtype)
        trainable_weights = tf.random.normal(shape=(trainable_weights,), mean=0, stddev=self.std, dtype=dtype)
        not_trainable_weights = tf.Variable(not_trainable_weights, trainable=False)
        trainable_weights = tf.Variable(trainable_weights, trainable=True)
        #tf.print(tf.reshape(tf.concat([not_trainable_weights, trainable_weights], axis=0), shape))
        return tf.reshape(tf.concat([not_trainable_weights, trainable_weights], axis=0), shape)