from tensorflow import keras
from tensorflow.keras import layers
import Representation
import tensorflow as tf
import random
import Activation
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from Optimizer import CustomAdam
import warnings
from Layer import DropConnect, NoisyReLU
from Initializer import FixedWeightInitializer
import os
import GPUtil


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class NeuralNetwork():
    def __init__(self, dataset, layers=None, neurons=None):
        self.dataset = dataset
        self.layers = layers
        self.neurons = neurons


    def set_config(self, shuffle, random_flip, random_rotation, random_zoom, random_translation, random_contrast, input_noise,
              label_smoothing, weight_std, dropout, drop_connect, drnn, activation_noise, loss_noise,
              optimizer, lr, lr_schedule, batch_size, batch_schedule, weight_noise, gradient_noise, gradient_dropout,
                   metric=keras.metrics.CategoricalAccuracy(), epochs=20, iterations=100000, patience=100000, verbose=0, batch_range=[16, 1024], lr_range=[1,5],
                   sleep=3, save_best=False, cut_threshold=0.4):
        self.activation_noise = activation_noise
        #print(self.activation_noise)
        self.loss_noise = loss_noise
        self.input_noise = input_noise
        if weight_noise >= 6:
            self.weight_noise = 0
        else:
            self.weight_noise = 1 / np.power(10, weight_noise)
        #print(self.weight_noise)
        self.metric = metric
        self.batch_size = batch_size
        self.epochs = epochs
        self.iterations = iterations
        self.patience = patience
        self.verbose = verbose
        self.dropout = dropout
        self.gradient_noise = gradient_noise
        self.gradient_dropout = gradient_dropout
        self.save_best = save_best
        self.optimizer = optimizer
        self.batch_increase = 0
        if (batch_schedule > 0):
            self.batch_increase = 1

        self.batch_schedule = int(((1-np.abs(batch_schedule)) * self.epochs)+1)
        #print(self.batch_increase, self.batch_schedule)

        self.lr_increase = 0
        if (lr_schedule > 0):
            self.lr_increase = 1

        self.lr_schedule = int(((1-np.abs(lr_schedule)) * self.epochs)+1)
        #print(self.lr_increase, self.lr_schedule)

        #self.lr_decay = int((lr_decay * self.epochs)+1)
        #print(self.double_batch_on)
        self.drop_connect = drop_connect
        self.drnn = drnn
        self.weight_std = weight_std

        self.label_smoothing = label_smoothing
        self.shuffle = shuffle
        self.lr = 1 / np.power(10, lr)
        self.random_flip = random_flip
        self.random_rotation = random_rotation
        self.random_zoom = random_zoom
        self.random_translation = random_translation
        self.random_contrast = random_contrast
        #self.max_batch = len(self.dataset.X_trainSampled)
        self.batch_range = batch_range
        self.sleep = sleep
        self.cut_threshold = cut_threshold
        self.lr_range = lr_range

        #print(self.lr_decay)
        #print()



    def fit_keras(self):
        self.optimizer = CustomAdam(gradient_dropout=self.gradient_dropout, gradient_noise=self.gradient_noise, weight_noise=self.weight_noise, drnn=self.drnn)
        #self.optimizer = FixedMaskAdam(self.drnn)
        #self.optimizer = tf.optimizers.Adam()

        time_callback = TimeHistory()

        self.model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=[self.metric])
        start = time.time()

        current_batch = self.batch_size
        current_epochs = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            while current_epochs+self.double_batch_on <= self.epochs:
                current_epochs += self.double_batch_on
                self.history = self.model.fit(self.dataset.X_trainSampled, self.dataset.y_trainSampled,
                                          validation_data=(self.dataset.X_val, self.dataset.y_val),
                                          epochs=self.double_batch_on, verbose=self.verbose, batch_size=current_batch, callbacks=[time_callback])
                #print(current_batch)
                current_batch *= 2
                if (current_batch >= self.max_batch):
                    current_batch = self.max_batch

            self.history = self.model.fit(self.dataset.X_trainSampled, self.dataset.y_trainSampled,
                                          validation_data=(self.dataset.X_val, self.dataset.y_val),
                                          epochs=self.epochs-current_epochs, verbose=self.verbose, batch_size=current_batch,
                                          callbacks=[time_callback])
        #print(time_callback.times)
        self.training_time = time.time() - start
        self.epoch_time = np.median(time_callback.times)
        start = time.time()
        score_val = self.model.evaluate(self.dataset.X_val, self.dataset.y_val, verbose=0)
        self.inference_time = time.time() - start
        start = time.time()
        score_test = self.model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=0)
        self.inference_time = time.time() - start

        self.val_score = score_val[1]
        self.test_score = score_test[1]

        self.val_loss = score_val[0]
        self.test_loss = score_test[0]


    def fit(self):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.dataset.X_trainSampled, self.dataset.y_trainSampled))
        if (self.shuffle):
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        else:
            train_dataset = train_dataset.batch(self.batch_size)


        # Prepare the validation dataset.
        if (self.dataset.training_split < 1.0):
            val_dataset = tf.data.Dataset.from_tensor_slices((self.dataset.X_val, self.dataset.y_val))
            val_dataset = val_dataset.batch(self.batch_size)

        # Instantiate an optimizer to train the model.

        decay_steps = 1000
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            self.lr, decay_steps, alpha=0.2)

        if (self.optimizer == 3):
            optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        elif (self.optimizer == 2):
            optimizer = tf.optimizers.SGD(learning_rate=self.lr, momentum=0.9)
        elif (self.optimizer <= 1):
            optimizer = tf.optimizers.SGD(learning_rate=self.lr)


        #self.drnn_mask = tf.cast(tf.random.uniform(shape=self.model.trainable_variables.shape) > self.drnn, dtype=tf.float32)


        # Prepare the metrics.
        train_acc_metric = self.metric
        val_acc_metric = self.metric


        loss_fn = self.loss_fn


        @tf.function
        def train_step(x, y, epoch):
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss_value = loss_fn(y, logits)
            #print(self.model.trainable_weights)

            #print(self.model.trainable_weights)


            grads = tape.gradient(loss_value, self.model.trainable_weights)

            #tf.print(loss_value)
            #tf.print(grads[0][0][0][0])
            #tf.print('grads before', grads)
            #tf.print(type(grads[0]))
            t = tf.cast(optimizer.iterations, grads[0].dtype)
            gradient_noise_std = tf.math.sqrt(tf.math.divide_no_nan(tf.cast(self.gradient_noise, tf.float32), tf.math.pow(t+1.0, 0.55)))
            #tf.print('gradient_noise_std', gradient_noise_std)
            weight_noise_std = tf.math.sqrt(tf.math.divide_no_nan(tf.cast(self.weight_noise, tf.float32), tf.math.pow(t+1.0, 0.55)))
            #tf.print('weight_noise_std', weight_noise_std)

            #tf.print(weight_noise_std)
            #tf.print('weights before', self.model.trainable_weights)
            #tf.print()
            #tf.print('grads before', grads)
            #tf.print()

            #tf.print('noise', gradient_noise_std)

            #print('grads', grads.shape)
            #print('mask', self.mask.shape)

            new_grads = []
            for x in grads:
                m = tf.cast(tf.random.uniform(shape=x.shape) > self.gradient_dropout, dtype=tf.float32)
                new_grads.append(x*m)

            grads = new_grads

            grads = [g * m for g, m in zip(grads, self.drnn_mask)]
            grads = [tf.math.add(x, tf.random.normal(tf.shape(x), 0, gradient_noise_std)) for x in grads]
            #grads = [tf.math.multiply(x, tf.cast(tf.random.uniform(shape=x.shape) > self.gradient_dropout, dtype=tf.float32)) for x in grads]



            #self.model.trainable_weights = [tf.math.add(x, tf.random.normal(tf.shape(x), 0, weight_noise_std)) for x in self.model.trainable_weights]
            #grads = grads * self.drnn_mask


            #tf.print('grads after', grads)
            #grads = map(self.gradient.s2, grads)
            #tf.print(grads[0][0][0][0])
            for var in self.model.trainable_variables:
            #    if len(layer.get_weights) > 0:
                    #tf.print(layer.name, layer.get_weights())
                    #tf.print(var)
                    noise = tf.random.normal(var.shape, 0, weight_noise_std)
                    var.assign_add(noise)
            #self.model.



            #tf.print('weights before', self.model.trainable_weights)


            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            #tf.print('grads after', grads)
            #tf.print()
            #tf.print('weights after', self.model.trainable_weights)
            #tf.print()

            #tf.print('weights after', self.model.trainable_weights)


            train_acc_metric.update_state(y, logits)
            return loss_value

        @tf.function
        def test_step(x, y):
            val_logits = self.model(x, training=False)
            val_acc_metric.update_state(y, val_logits)

        import time

        best_val = 0
        train_step_counter = 1
        self.model.compile(loss=loss_fn, metrics=[train_acc_metric])

        self.drnn_mask = []

        for x in self.model.trainable_weights:

            #print(x.shape)

            self.drnn_mask.append(tf.cast(tf.random.uniform(shape=x.shape) > self.drnn, dtype=tf.float32))

        patience_counter = 0
        current_batch = self.batch_size
        current_lr = self.lr

        gpu_timer = time.time()

        start = time.time()




        # Define the data augmentation pipeline
        if (self.random_flip <= 1):

            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.RandomRotation(self.random_rotation),
                tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=self.random_zoom, width_factor=self.random_zoom),
                tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=self.random_translation, width_factor=self.random_translation),
                tf.keras.layers.experimental.preprocessing.RandomContrast(factor=self.random_contrast),
            ])

        elif (self.random_flip == 2):

            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(self.random_rotation),
                tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=self.random_zoom, width_factor=self.random_zoom),
                tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=self.random_translation,
                                                                             width_factor=self.random_translation),
                tf.keras.layers.experimental.preprocessing.RandomContrast(factor=self.random_contrast),
            ])

        elif (self.random_flip == 3):

            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(self.random_rotation),
                tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=self.random_zoom, width_factor=self.random_zoom),
                tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=self.random_translation,
                                                                             width_factor=self.random_translation),
                tf.keras.layers.experimental.preprocessing.RandomContrast(factor=self.random_contrast),
            ])

        elif (self.random_flip == 4):

            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.RandomFlip("vertical"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(self.random_rotation),
                tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=self.random_zoom, width_factor=self.random_zoom),
                tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=self.random_translation,
                                                                             width_factor=self.random_translation),
                tf.keras.layers.experimental.preprocessing.RandomContrast(factor=self.random_contrast),
            ])


        for epoch in range(self.epochs):
            #print(optimizer.lr, optimizer.learning_rate, current_lr)
            average_train_step_time = 0
            train_step_counter_within_epoch = 0
            #if (self.adaptive_batch and patience_counter == 0):
            #    current_batch = self.batch_size
            #if (self.adaptive_batch and patience_counter != 0):
            #        current_batch += 4

            if (self.batch_increase>0 and epoch > 0 and epoch%self.batch_schedule==0):
                current_batch *= 2

            if (self.batch_increase==0 and epoch > 0 and epoch % self.batch_schedule == 0):
                current_batch = int(current_batch /2)

            if (current_batch > self.batch_range[1]):
                    current_batch = self.batch_range[1]

            if (current_batch < self.batch_range[0]):
                    current_batch = self.batch_range[0]

                    # self.sleep = self.sleep/2
            #print(self.lr_schedule, epoch % self.lr_schedule)
            if (self.lr_increase>0 and epoch > 0 and epoch % self.lr_schedule == 0):
                    current_lr *= 2
                    #print('increase')

            if (self.lr_increase== 0 and epoch > 0 and epoch % self.lr_schedule == 0):
                    current_lr = current_lr/2
                    #print('decrease')

                    #print('now now', current_lr, self.min_lr)
                    #print('check', current_lr < self.min_lr)
            if (current_lr > 1 / np.power(10, self.lr_range[0])):
                    current_lr = 1 / np.power(10, self.lr_range[0])

            if (current_lr < 1 / np.power(10, self.lr_range[1])):
                        current_lr = 1 / np.power(10, self.lr_range[1])

            #print('after', optimizer.lr, optimizer.learning_rate, current_lr)



            optimizer.lr.assign(current_lr)


                #self.sleep *= 2



            #print(self.sleep)
            number_of_training_steps_per_epoch = len(self.dataset.X_trainSampled)//current_batch


            train_dataset = tf.data.Dataset.from_tensor_slices(
                (self.dataset.X_trainSampled, self.dataset.y_trainSampled))
            if (self.shuffle):
                train_dataset = train_dataset.shuffle(buffer_size=1024).batch(current_batch)
            else:
                train_dataset = train_dataset.batch(current_batch)

            #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(current_batch)

            #Prepare the validation dataset.
            if (self.dataset.training_split < 1.0):
                val_dataset = tf.data.Dataset.from_tensor_slices((self.dataset.X_val, self.dataset.y_val))
                val_dataset = val_dataset.batch(current_batch)
            if (self.verbose):
                print("\nStart of epoch %d" % (epoch,))
                print('cuurent batch:', current_batch)
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                #print((epoch), (train_step_counter))
                #print(type(epoch), type(train_step_counter))
                #optimizer.lr = lr_decayed_fn(optimizer.lr)
                #print(optimizer.lr)

                timer = time.time()
                images = data_augmentation(x_batch_train)
                loss_value = train_step(images, y_batch_train, epoch)
                average_train_step_time += time.time()-timer


                '''
                #if (time.time()-gpu_timer>1):

                #    temp = int(os.popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader").read())
                #    sleep_step = 0.1
                #    utilization = GPUtil.getGPUs()[0].load
                #    print(utilization)
                #    while (utilization > 0.6):
                #        time.sleep(sleep_step)
                #        utilization = GPUtil.getGPUs()[0].load
                #        print(utilization)
                    #while temp > 60:
                    #    print(temp)
                    #    time.sleep(sleep_step)
                    #    sleep_step *= 2
                    #    temp = int(os.popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader").read())
                    gpu_timer = time.time()
                '''


                train_step_counter += 1
                train_step_counter_within_epoch += 1

                total_active_time = (average_train_step_time/train_step_counter_within_epoch) * number_of_training_steps_per_epoch

                #if train_step_counter%10 == 0 :

                #print(average_train_step_time, number_of_training_steps_per_epoch, total_active_time)

                #print((self.sleep-total_active_time)/number_of_training_steps_per_epoch)

                #print(max(0.0, (self.sleep-total_active_time)/number_of_training_steps_per_epoch))

                time.sleep(max(0.0, (self.sleep-total_active_time)/number_of_training_steps_per_epoch))

            #print(train_step_counter)

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            if (self.verbose):
                print("Training acc: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            if (self.dataset.training_split < 1.0):
                for x_batch_val, y_batch_val in val_dataset:
                    test_step(x_batch_val, y_batch_val)
                val_acc = val_acc_metric.result()
                val_acc_metric.reset_states()

            patience_counter += 1

            if (self.verbose):
                if (self.dataset.training_split < 1.0):
                    print("Validation acc: %.4f" % (float(val_acc),))
                print("Time taken: %.2fs" % (time.time() - start_time))
                print("Patience: ", patience_counter)

            if (self.dataset.training_split < 1.0):
                if (val_acc > best_val):
                    best_val = val_acc

                    self.model.save_weights('best_model.h5')
                    patience_counter = 0

            if (self.dataset.training_split < 1.0 and epoch == 0 and val_acc < self.cut_threshold):
                break

            if (optimizer.iterations > self.iterations):
                break

            if (patience_counter == self.patience):
                break


        if(self.save_best):
            self.model.load_weights('best_model.h5')

        self.training_time = time.time() - start
        self.epoch_time = self.training_time / (self.epochs + 1)

        if (self.dataset.training_split < 1.0):
            score_val = self.model.evaluate(self.dataset.X_val, self.dataset.y_val, verbose=0)
            self.val_score = score_val[1]
            self.val_loss = score_val[0]


        start = time.time()
        score_test = self.model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=0)
        self.inference_time = time.time() - start
        self.test_score = score_test[1]
        self.test_loss = score_test[0]

    def loss_fn(self, y_true, y_hat):
        l = tf.keras.losses.categorical_crossentropy(y_true, y_hat, label_smoothing=self.label_smoothing)
        return  l * tf.random.normal(tf.shape(l), 1.0, self.loss_noise)

    def activation_fn(self, x):
        return tf.nn.relu(x) + np.random.normal(0.0, self.activation_noise)


    def create_model(self):
        inputs = keras.layers.InputLayer(input_shape=self.dataset.inputShape)
        x = tf.keras.layers.GaussianNoise(self.input_noise)(inputs)
        x = layers.Conv2D(2, kernel_size=(3, 3))(x)
        x = self.activation()(x)
        x = layers.Conv2D(2, kernel_size=(3, 3))(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(2, kernel_size=(3, 3))(inputs)
        x = self.activation()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(self.dataset.num_classes, activation="softmax")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)





class VGG1(NeuralNetwork):
    def create_model(self):
            inputs = tf.keras.Input(shape=self.dataset.input_shape)
            x = tf.keras.layers.GaussianNoise(self.input_noise)(inputs)
            x = layers.Conv2D(32, kernel_size=(3, 3), activation=self.activation_fn)(x)
            x = layers.Conv2D(32, kernel_size=(3, 3), activation=self.activation_fn)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Dropout(self.dropout)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(128, activation=self.activation_fn)(x)
            x = layers.Dropout(self.dropout)(x)
            outputs = layers.Dense(self.dataset.num_classes, activation="softmax")(x)
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

class VGG(NeuralNetwork):
    def __init__(self, dataset, blocks):
        self.dataset =  dataset
        self.blocks = blocks

    def create_model(self):

        inputs = tf.keras.Input(shape=self.dataset.input_shape)
        x = tf.keras.layers.GaussianNoise(self.input_noise)(inputs)
        filter_size = 32
        x = DropConnect(layers.Conv2D(filter_size, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = DropConnect(layers.Conv2D(filter_size, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
        x = layers.Dropout(self.dropout)(x)
        for _ in range(self.blocks-1):
            if (filter_size < 512):
                filter_size *= 2
            x = DropConnect(layers.Conv2D(filter_size, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), padding="same"), prob=self.drop_connect)(x)
            x = NoisyReLU(stddev=self.activation_noise)(x)
            x = DropConnect(layers.Conv2D(filter_size, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), padding="same"), prob=self.drop_connect)(x)
            x = NoisyReLU(stddev=self.activation_noise)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
            x = layers.Dropout(self.dropout)(x)
        x = layers.Flatten()(x)
        x = DropConnect(layers.Dense(128, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std)), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(self.dataset.num_classes, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), activation="softmax")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)



class VGG16(NeuralNetwork):
    def __init__(self, dataset):
        self.dataset =  dataset

    def create_model(self):
        inputs = tf.keras.Input(shape=self.dataset.input_shape)
        x = tf.keras.layers.GaussianNoise(self.input_noise)(inputs)
        x = DropConnect(layers.Conv2D(64, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = DropConnect(layers.Conv2D(64, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
        #x = layers.Dropout(self.dropout)(x)

        x = DropConnect(layers.Conv2D(128, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = DropConnect(layers.Conv2D(128, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
        #x = layers.Dropout(self.dropout)(x)

        x = DropConnect(layers.Conv2D(256, kernel_size=(3, 3),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=self.weight_std),
                                      padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = DropConnect(layers.Conv2D(256, kernel_size=(3, 3),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=self.weight_std),
                                      padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = DropConnect(layers.Conv2D(256, kernel_size=(3, 3),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=self.weight_std),
                                      padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        x = DropConnect(layers.Conv2D(512, kernel_size=(3, 3),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=self.weight_std),
                                      padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = DropConnect(layers.Conv2D(512, kernel_size=(3, 3),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=self.weight_std),
                                      padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = DropConnect(layers.Conv2D(512, kernel_size=(3, 3),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=self.weight_std),
                                      padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        x = DropConnect(layers.Conv2D(512, kernel_size=(3, 3),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=self.weight_std),
                                      padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = DropConnect(layers.Conv2D(512, kernel_size=(3, 3),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=self.weight_std),
                                      padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = DropConnect(layers.Conv2D(512, kernel_size=(3, 3),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=self.weight_std),
                                      padding="same"), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)



        x = layers.Flatten()(x)
        x = DropConnect(layers.Dense(4096, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std)), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = layers.Dropout(self.dropout)(x)
        x = DropConnect(
            layers.Dense(4096, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std)),
            prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(self.dataset.num_classes, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), activation="softmax")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

class FC(NeuralNetwork):
    def __init__(self, dataset, layers, neurons):
        self.dataset =  dataset
        self.layers = layers
        self.neurons = neurons


    def create_model(self):
        inputs = tf.keras.Input(shape=self.dataset.input_shape)
        x = layers.Flatten()(inputs)
        x = tf.keras.layers.GaussianNoise(self.input_noise)(x)
        x = DropConnect(layers.Dense(self.neurons, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std)), prob=self.drop_connect)(x)
        x = NoisyReLU(stddev=self.activation_noise)(x)
        x = layers.Dropout(self.dropout)(x)
        for _ in range(self.layers-1):
                x = DropConnect(layers.Dense(self.neurons, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std)), prob=self.drop_connect)(x)
                x = NoisyReLU(stddev=self.activation_noise)(x)
                x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(self.dataset.num_classes, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.weight_std), activation="softmax")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)



