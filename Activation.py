from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from Representation import  Function, PAU
import tensorflow as tf
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Pade_Activation_Unit(keras.layers.Layer):

    def __init__(self, coefficients=None, **kwargs):
        super(Pade_Activation_Unit, self).__init__(**kwargs)
        if coefficients is None:
            coefficients = [0.02979246, 0.61837738, 2.32335207, 3.05202660, 1.48548002, 0.25103717, 1.14201226,
                            4.39322834, 0.87154450, 0.34720652]
        #print('inside', coefficients)
        self.object = PAU(coefficients)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._a0 = self.add_weight(name='a0',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=self.object.coefficients[0]),
                                   trainable=True)
        self._a1 = self.add_weight(name='a1',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=self.object.coefficients[1]),
                                   trainable=True)
        self._a2 = self.add_weight(name='a2',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=self.object.coefficients[2]),
                                   trainable=True)
        self._a3 = self.add_weight(name='a3',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=self.object.coefficients[3]),
                                   trainable=True)
        self._a4 = self.add_weight(name='a4',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=self.object.coefficients[4]),
                                   trainable=True)
        self._a5 = self.add_weight(name='a5',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=self.object.coefficients[5]),
                                   trainable=True)
        self._b1 = self.add_weight(name='b1',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=self.object.coefficients[6]),
                                   trainable=True)
        self._b2 = self.add_weight(name='b2',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=self.object.coefficients[7]),
                                   trainable=True)
        self._b3 = self.add_weight(name='b3',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=self.object.coefficients[8]),
                                   trainable=True)
        self._b4 = self.add_weight(name='b4',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=self.object.coefficients[9]),
                                   trainable=True)

        super(Pade_Activation_Unit, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        '''
        numerator = tf.convert_to_tensor([self._a0, tf.math.multiply(self._a1, x),
                                          tf.math.multiply(self._a2, tf.math.pow(x, 2)),
                                          tf.math.multiply(self._a3, tf.math.pow(x, 3)),
                                          tf.math.multiply(self._a4, tf.math.pow(x, 4)),
                                          tf.math.multiply(self._a5, tf.math.pow(x, 5))])
        denominator = tf.convert_to_tensor([tf.constant(1.0), tf.math.multiply(self._b1, x),
                                            tf.math.multiply(self._b2, tf.math.pow(x, 2)),
                                            tf.math.multiply(self._b3, tf.math.pow(x, 3)),
                                            tf.math.multiply(self._b4, tf.math.pow(x, 4))])

        numerator = tf.cast(tf.math.reduce_sum(numerator), tf.float32)
        denominator = tf.cast(tf.math.reduce_sum(denominator), tf.float32)

        return tf.math.divide_no_nan(numerator, denominator)
        '''
        numerator = self._a0 + tf.math.multiply(self._a1, x) + tf.math.multiply(self._a2,
                                                                                tf.math.pow(x, 2)) + tf.math.multiply(
            self._a3, tf.math.pow(x, 3)) + tf.math.multiply(self._a4, tf.math.pow(x, 4)) + tf.math.multiply(self._a5,
                                                                                                            tf.math.pow(
                                                                                                                x, 5))

        denominator = 1.0 + tf.math.multiply(self._b1, x) + tf.math.multiply(self._b2,
                                                                             tf.math.pow(x, 2)) + tf.math.multiply(
            self._b3, tf.math.pow(x, 3)) + tf.math.multiply(self._b4, tf.math.pow(x, 4))
        #tf.print('a0', self._a0)
        #tf.print('a1', self._a1)
        return tf.math.divide_no_nan(numerator, denominator)
        # print(
        # return tf.math.multiply(self._x1, x) + tf.math.multiply(self._x2, tf.math.pow(x, 2))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class PANGAEA_Activation(keras.layers.Layer):

    def __init__(self, function, trainable=True, **kwargs):
        super(PANGAEA_Activation, self).__init__(**kwargs)
        self.function = Function(function, tf.ones((9,)))
        self.trainable = trainable

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._a1 = self.add_weight(name='a1',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=1.0),
                                   trainable=self.trainable)
        self._a2 = self.add_weight(name='a2',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=1.0),
                                   trainable=self.trainable)
        self._a3 = self.add_weight(name='a3',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=1.0),
                                   trainable=self.trainable)
        self._a4 = self.add_weight(name='a4',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=1.0),
                                   trainable=self.trainable)
        self._a5 = self.add_weight(name='a5',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=1.0),
                                   trainable=self.trainable)
        self._a6 = self.add_weight(name='a6',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=1.0),
                                   trainable=self.trainable)
        self._a7 = self.add_weight(name='a7',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=1.0),
                                   trainable=self.trainable)
        self._a8 = self.add_weight(name='a8',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=1.0),
                                   trainable=self.trainable)
        self._a9 = self.add_weight(name='a9',
                                   shape=(1,),
                                   initializer=tf.keras.initializers.Constant(value=1.0),
                                   trainable=self.trainable)

        super(PANGAEA_Activation, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        self.function.weights = [self._a1, self._a2, self._a3, self._a4, self._a5, self._a6, self._a7, self._a8,
                                 self._a9]
        #tf.print(self.function.weights)
        #tf.print(self.function.operators)
        #tf.print(self.function.s2(1.0))
        return self.function.calculate(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class MyLoss(keras.layers.Layer):
    def __init__(self, var1, var2):
        super(MyLoss, self).__init__()
        self.var1 = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True)
        self.var2 = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True)

    def get_vars(self):
        return self.var1, self.var2

    def custom_loss(self, y_true, y_pred):
        return self.var1 * (tf.math.abs(y_true - y_pred)) + self.var2 ** 2

    def call(self, y_true, y_pred):
        self.add_loss(self.custom_loss(y_true, y_pred))
        return y_pred