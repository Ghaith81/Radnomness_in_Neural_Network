import numpy as np
import tensorflow as tf
import sys
from numpy import random

def zero(x):
    return x-x

def one(x):
    return x-x+1

def x(x):
    return x

def x_2(x):
    return tf.math.pow(x, 2)

def x_3(x):
    return tf.math.pow(x, 3)

def sqrt(x):
    # return tf.cast(tf.math.sqrt(x), dtype=float)
    return tf.math.sqrt(x)

def exp(x):
    # return tf.cast(tf.math.exp(x), dtype=tf.float32)
    return tf.math.exp(x)

def e_n_x_2(x):
    # return tf.cast(tf.math.exp(-1 * np.power(x, 2)), dtype=tf.float32)
    return tf.math.exp(-1 * x**2)
    #return np.power(math.e, -1 * np.power(x, 2))

def log_1_e_x(x):
    return tf.math.log(1 + tf.math.exp(x))

def log(x):
    return tf.math.log(tf.math.abs(x) + sys.float_info.epsilon)

def max0(x):
    return tf.math.maximum(0.0, x)

def min0(x):
    return tf.math.minimum(0.0, x)

def dev(x1, x2):
    return x1/(x2 + sys.float_info.epsilon)

def sinc(x):
    x = tf.where(tf.abs(x) < 1e-20, 1e-20 * tf.ones_like(x), x)
    return tf.sin(x) / x

def randomf(x):
    #return tf.random.uniform(shape=[], minval=0, maxval=tf.math.multiply(0.1, x), dtype=tf.float32)+0.0
    return random.uniform(0.0, 0.1) + 0.0
operators = {
    1: tf.math.add,
    2: tf.math.subtract,
    3: tf.math.multiply,
    4: tf.math.divide_no_nan,
    5: tf.math.maximum,
    6: tf.math.minimum,
    7: zero,
    8: one,
    9: tf.identity,
    10: tf.math.negative,
    11: tf.math.abs,
    12: x_2,
    13: x_3,
    14: tf.math.sqrt,
    15: exp,
    16: e_n_x_2,
    17: log_1_e_x,
    18: log,
    19: tf.math.sin,
    20: tf.math.sinh,
    21: tf.math.asinh,
    22: tf.math.cos,
    23: tf.math.cosh,
    24: tf.math.tanh,
    25: tf.math.atanh,
    26: tf.math.sigmoid,
    27: tf.math.erf,
    28: sinc,
    29: max0,
    30: min0,
    31: randomf
}

class Maxout():
    '''
    def __init__(self, weights):
        self.weights = np.array(weights).reshape(-1, 2)
     '''

    def __init__(self, weights):
        self.weights = np.array(weights)
        self.xs = np.linspace(-1, 1, num=self.weights.shape[0])


    def linear_func(self, a, b, x):
        return (a * x) + b

    '''
    def s2(self, x):
        funcs = []
        for weight in self.weights:
            funcs.append(self.linear_func(weight[0], weight[1], x))
        return tf.math.reduce_max(funcs,  axis=0)

    '''

    def s2(self, x):
        value = tf.math.maximum(0.0, x)
        #value = 0.0
        i = 0
        while i < len(self.weights):
            value += tf.math.multiply(tf.cast(self.weights[i], dtype=tf.float32), tf.math.maximum(0.0, tf.math.add(tf.math.negative(x), self.xs[i])))
            i += 1
        #value += tf.math.multiply(tf.cast(tf.math.negative(self.weights[i-1]), dtype=tf.float32), tf.math.maximum(0.0, tf.math.add(x, tf.cast(tf.math.negative(self.xs[i-1]), dtype=tf.float32))))
        #while i < len(self.weights):
        #    value += tf.math.multiply(tf.cast(self.weights[i], dtype=tf.float32), tf.math.maximum(0.0, tf.math.add(x, self.xs[i])))
        #    i += 1

        #for weight in self.weights:
        #    value +=weight[0]*tf.math.maximum(0.0, self.linear_func(-1.0, weight[1], x))
        #for weight in self.weights[int(len(self.weights)/2):]:
        #    value +=weight[0]*tf.math.maximum(0.0, self.linear_func(1, weight[1], x))
        return value


    '''
    def __init__(self, y):
        self.ys = np.array(y)
        self.xs = np.linspace(-1, 1, num=self.ys.shape[0])
        weights = []
        for i in range(self.xs.shape[0] - 1):
            weights.extend(self.slope_intercept(self.xs[i], self.ys[i], self.xs[i + 1], self.ys[i + 1]))
        self.weights = tf.convert_to_tensor(np.array(weights).reshape(-1, 2))


    def linear_func(self, a, b, x):
        return a*x + b

    def slope_intercept(self, x1, y1, x2, y2):
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return [a, b]

    def s2(self, x):
        i = 0
        while i < len(self.weights)-1:
            lower_tensor = tf.greater(x, self.xs[i])
            upper_tensor = tf.less(x, self.xs[i+1])
            in_range = tf.logical_and(lower_tensor, upper_tensor)

            result = tf.cond(in_range, lambda: self.weights[i][0]*x +self.weights[i][1], lambda: self.weights[i][0]*x +self.weights[i][1])
            return result
            #t = (tf.math.greater(x, tf.constant(y, dtype=float)))
            #rank = tf.cond(t, lambda:rank+1, lambda:rank+0)
            #rank = tf.cast(rank, dtype=tf.int32)
            #return self.linear_func(self.weights[i][0], self.weights[i][1], x)
    '''
class PAU():
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def calculate(self, x):
        numerator = tf.convert_to_tensor([tf.fill(tf.shape(x), self.coefficients[0]), tf.math.multiply(self.coefficients[1], x), tf.math.multiply(self.coefficients[2], tf.math.pow(x, 2)), tf.math.multiply(self.coefficients[3], tf.math.pow(x, 3)), tf.math.multiply(self.coefficients[4], tf.math.pow(x, 4)), tf.math.multiply(self.coefficients[5], tf.math.pow(x, 5))])
        denominator = tf.convert_to_tensor([tf.fill(tf.shape(x), tf.constant(1.0)), tf.math.multiply(self.coefficients[6], x), tf.math.multiply(self.coefficients[7],tf.math.pow(x, 2)), tf.math.multiply(self.coefficients[8], tf.math.pow(x, 3)), tf.math.multiply(self.coefficients[9], tf.math.pow(x, 4))])

        numerator = tf.cast(tf.math.reduce_sum(numerator, axis=0), tf.float32)
        denominator = tf.cast(tf.math.reduce_sum(denominator, axis=0), tf.float32)

        return tf.math.divide_no_nan(numerator, denominator)

class Taylor():
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def calculate(self, x):
        series = []
        idx = 0
        for coef in self.coefficients:
            series.append(tf.math.multiply(coef, tf.math.pow(x, idx+1)))
            idx+=1
        series = tf.convert_to_tensor(series)
        #series = tf.convert_to_tensor([tf.math.multiply(self.coefficients[0], x), tf.math.multiply(self.coefficients[1], tf.math.pow(x, 2)), tf.math.multiply(self.coefficients[2], tf.math.pow(x, 3)), tf.math.multiply(self.coefficients[3], tf.math.pow(x, 4)), tf.math.multiply(self.coefficients[4], tf.math.pow(x, 5)), tf.math.multiply(self.coefficients[5], tf.math.pow(x, 6)), tf.math.multiply(self.coefficients[6], tf.math.pow(x, 7)), tf.math.multiply(self.coefficients[7], tf.math.pow(x, 8)), tf.math.multiply(self.coefficients[8], tf.math.pow(x, 9))])
        series = tf.cast(tf.math.reduce_sum(series), tf.float32)
        return series

class Function():
    def __init__(self, operators, weights=tf.ones((9,))):
        self.operators = operators
        self.weights = weights
        #print(self.operators, self.weights)

    def s1(self, x):
        return BinaryOperator(self.operators[0], self.weights[0],
                              UnaryOperator(self.operators[1], self.weights[1], x),
                              UnaryOperator(self.operators[2], self.weights[2], x)).calculate()

    def calculate(self, x):
        return BinaryOperator(self.operators[0], self.weights[0],
                              UnaryOperator(self.operators[1], self.weights[1],
                                            BinaryOperator(self.operators[3], self.weights[3],
                                                           UnaryOperator(self.operators[5], self.weights[5], x),
                                                           UnaryOperator(self.operators[6], self.weights[6], x))),
                              UnaryOperator(self.operators[2], self.weights[2],
                                            BinaryOperator(self.operators[4], self.weights[4],
                                                           UnaryOperator(self.operators[7], self.weights[7], x),
                                                           UnaryOperator(self.operators[8], self.weights[8], x)))).calculate()

    def sl(self, x):
        return UnaryOperator(self.operators[1],
                             BinaryOperator(self.operators[3],
                                            UnaryOperator(self.operators[5], x),
                                            UnaryOperator(self.operators[6], x))).calculate()

    def sr(self, x):
        return UnaryOperator(self.operators[2],
                             BinaryOperator(self.operators[4],
                                            UnaryOperator(self.operators[7], x),
                                            UnaryOperator(self.operators[8], x))).calculate()


class BinaryOperator():
    def __init__(self, operator, weight, x1, x2):
        self.operator = operators[operator]
        self.weight = weight
        self.x1 = x1
        self.x2 = x2

    def calculate(self):
        if ((isinstance(self.x1, UnaryOperator) or isinstance(self.x1, BinaryOperator)) and
                (isinstance(self.x2, UnaryOperator) or isinstance(self.x2, BinaryOperator))):
            return self.operator(self.x1.calculate(), self.x2.calculate())
        elif ((isinstance(self.x1, UnaryOperator) or isinstance(self.x1, BinaryOperator)) and
              (not isinstance(self.x2, UnaryOperator) and not isinstance(self.x2, BinaryOperator))):
            return self.operator(self.x1.calculate(), self.x2)
        elif ((not isinstance(self.x1, UnaryOperator) and not isinstance(self.x1, BinaryOperator)) and
              (isinstance(self.x2, UnaryOperator) or isinstance(self.x2, BinaryOperator))):
            return tf.math.multiply(self.weight, self.operator(self.x1, self.x2.calculate()))
        else:
            return tf.math.multiply(self.weight, self.operator(self.x1, self.x2))


class UnaryOperator():
    def __init__(self, operator, weight, x1):
        self.operator = operators[operator]
        self.weight = weight
        self.x1 = x1

    def calculate(self):
        if (isinstance(self.x1, UnaryOperator) or isinstance(self.x1, BinaryOperator)):
            return tf.math.multiply(self.weight, self.operator(self.x1.calculate()))
        else:
            return tf.math.multiply(self.weight, self.operator(self.x1))


class Leaf():
    def __init__(self, operator, weight, x, y):
        self.operator = operator
        self.weight = weight
        self.x = x
        self.y = y

    def leafCalculate(self):
        if (self.operator == 0):
            return 1.0
        elif (self.operator == 1):
            return -1.0
        elif (self.operator == 2):
            return self.x
        elif (self.operator == 3):
            return self.y
