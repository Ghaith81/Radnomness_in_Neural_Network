import tensorflow as tf

class CustomAdam(tf.keras.optimizers.Optimizer):
    def __init__(self, name='CustomAdam', learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                 gradient_dropout=0.0, gradient_noise=0.0, weight_noise=0.0, drnn=0.0, **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('epsilon', epsilon)
        self._set_hyper('gradient_dropout', gradient_dropout)
        self._set_hyper('gradient_noise', gradient_noise)
        self._set_hyper('weight_noise', weight_noise)
        self._set_hyper('drnn', drnn)
        self.mask = []


    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')


    def _resource_apply(self, grad, var, indices=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = self._get_hyper('epsilon', var_dtype)
        gradient_dropout = self._get_hyper('gradient_dropout', var_dtype)
        gradient_noise = self._get_hyper('gradient_noise', var_dtype)
        weight_noise = self._get_hyper('weight_noise', var_dtype)



        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')



        t = tf.cast(self.iterations, tf.float32)
        weight_noise_std = tf.math.sqrt(tf.math.divide_no_nan(weight_noise, tf.math.pow(t + 1.0, 0.55)))
        noise = tf.random.normal(shape=var.shape, mean=1, stddev=tf.math.sqrt(weight_noise_std), dtype=var_dtype)
        var.assign(var * noise)

        gradient_mask = tf.cast(tf.random.uniform(shape=var.shape) > gradient_dropout, dtype=tf.float32)
        gradient_noise_std = tf.math.sqrt(tf.math.divide_no_nan(gradient_noise, tf.math.pow(t + 1.0, 0.55)))

        grad = grad + tf.random.normal(tf.shape(grad), 0, gradient_noise_std)
        grad = grad * gradient_mask

        m_t = tf.cast(beta_1_t * m + (1 - beta_1_t) * grad, var_dtype)
        v_t = tf.cast(beta_2_t * v + (1 - beta_2_t) * tf.square(grad), var_dtype)
        var_update = var.assign_sub(lr_t * m_t / (tf.sqrt(v_t) + epsilon_t))

        return tf.group(*[var_update, m.assign(m_t), v.assign(v_t)])

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = self._get_hyper('epsilon', var_dtype)
        gradient_dropout = self._get_hyper('gradient_dropout', var_dtype)
        gradient_noise = self._get_hyper('gradient_noise', var_dtype)
        weight_noise = self._get_hyper('weight_noise', var_dtype)


        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        t = tf.cast(self.iterations, tf.float32)
        weight_noise_std = tf.math.sqrt(tf.math.divide_no_nan(weight_noise, tf.math.pow(t + 1.0, 0.55)))
        noise = tf.random.normal(shape=var.shape, mean=1, stddev=tf.math.sqrt(weight_noise_std), dtype=var_dtype)
        var.assign(var * noise)

        #tf.print('iteration', t)

        gradient_mask = tf.cast(tf.random.uniform(shape=var.shape) > gradient_dropout, dtype=tf.float32)
        gradient_noise_std = tf.math.sqrt(tf.math.divide_no_nan(gradient_noise, tf.math.pow(t + 1.0, 0.55)))

        grad = grad + tf.random.normal(tf.shape(grad), 0, gradient_noise_std)
        grad = grad * gradient_mask

        m_t = tf.cast(beta_1_t * m + (1 - beta_1_t) * grad, var_dtype)
        v_t = tf.cast(beta_2_t * v + (1 - beta_2_t) * tf.square(grad), var_dtype)
        var_update = var.assign_sub(lr_t * m_t / (tf.sqrt(v_t) + epsilon_t))

        return tf.group(*[var_update, m.assign(m_t), v.assign(v_t)])






