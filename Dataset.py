import numpy as np
from tensorflow import keras
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
import os
from sklearn.datasets import make_circles, make_moons
from matplotlib import cm

class Dataset():
    def __init__(self, X_train, y_train, X_test, y_test, num_classes, input_shape, expand=True, label_noise=0, training_split=1.0):
        self.num_classes =  num_classes
        self.input_shape = input_shape
        self.training_split = training_split

        # Scale images to the [0, 1] range
        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        if (expand):
            X_train = np.expand_dims(X_train, -1)
            X_test = np.expand_dims(X_test, -1)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if (self.training_split < 1.0):
            split = int(np.round(len(X_train) * self.training_split))
            self.X_train, self.X_val = self.X_train[:split, :], self.X_train[split:, :]
            self.y_train, self.y_val = self.y_train[:split], self.y_train[split:]


        if (label_noise):
            y_train_noisy = []
            for label in self.y_train:
                if random.random() < label_noise:
                    possible_labels = list(np.unique(y_train))
                    possible_labels.remove(label)
                    # print(label)
                    label = random.choice(possible_labels)
                    # print(label)
                y_train_noisy.append([label])
            self.y_train = y_train_noisy


        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        if (self.training_split < 1.0):
            self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)

        self.set_training_sample(self.X_train.shape[0])


    def set_training_sample(self, sample_size):
        self.X_trainSampled = self.X_train[:sample_size]
        self.y_trainSampled = self.y_train[:sample_size]


class MNIST(Dataset):
    def __init__(self, label_noise=0, training_split=1.0):
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        self.num_classes =  10
        self.input_shape = (28, 28, 1)
        self.training_split = training_split

        # Scale images to the [0, 1] range
        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)

        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

        self.expand = True



        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if (self.training_split < 1.0):
            split = int(np.round(len(X_train) * self.training_split))
            self.X_train, self.X_val = self.X_train[:split, :], self.X_train[split:, :]
            self.y_train, self.y_val = self.y_train[:split], self.y_train[split:]


        if (label_noise):
            y_train_noisy = []
            for label in self.y_train:
                if random.random() < label_noise:
                    possible_labels = list(np.unique(y_train))
                    possible_labels.remove(label)
                    # print(label)
                    label = random.choice(possible_labels)
                    # print(label)
                y_train_noisy.append([label])
            self.y_train = y_train_noisy


        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        if (self.training_split < 1.0):
            self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)

        self.set_training_sample(self.X_train.shape[0])

class Fashion_MNIST(Dataset):
    def __init__(self, label_noise=0, training_split=1.0):
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
        self.num_classes =  10
        self.input_shape = (28, 28, 1)
        self.training_split = training_split

        # Scale images to the [0, 1] range
        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)

        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

        self.expand = True




        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if (self.training_split < 1.0):
            split = int(np.round(len(X_train) * self.training_split))
            self.X_train, self.X_val = self.X_train[:split, :], self.X_train[split:, :]
            self.y_train, self.y_val = self.y_train[:split], self.y_train[split:]


        if (label_noise):
            y_train_noisy = []
            for label in self.y_train:
                if random.random() < label_noise:
                    possible_labels = list(np.unique(y_train))
                    possible_labels.remove(label)
                    # print(label)
                    label = random.choice(possible_labels)
                    # print(label)
                y_train_noisy.append([label])
            self.y_train = y_train_noisy


        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        if (self.training_split < 1.0):
            self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)

        self.set_training_sample(self.X_train.shape[0])

class CIFAR10(Dataset):
    def __init__(self, label_noise=0, training_split=1.0):
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        self.num_classes =  10
        self.input_shape = (32, 32, 3)
        self.training_split = training_split

        # Scale images to the [0, 1] range
        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255

        #X_train = np.expand_dims(X_train, -1)
        #X_test = np.expand_dims(X_test, -1)
        self.expand = False

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if (self.training_split < 1.0):
            split = int(np.round(len(X_train) * self.training_split))
            self.X_train, self.X_val = self.X_train[:split, :], self.X_train[split:, :]
            self.y_train, self.y_val = self.y_train[:split], self.y_train[split:]


        if (label_noise):
            y_train_noisy = []
            for label in self.y_train:
                if random.random() < label_noise:
                    possible_labels = list(np.unique(y_train))
                    possible_labels.remove(label)
                    # print(label)
                    label = random.choice(possible_labels)
                    # print(label)
                y_train_noisy.append([label])
            self.y_train = y_train_noisy


        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        if (self.training_split < 1.0):
            self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)

        self.set_training_sample(self.X_train.shape[0])

class CIFAR100(Dataset):
    def __init__(self, label_noise=0, training_split=1.0):
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
        self.num_classes =  100
        self.input_shape = (32, 32, 3)
        self.training_split = training_split

        # Scale images to the [0, 1] range
        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255

        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)


        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if (self.training_split < 1.0):
            split = int(np.round(len(X_train) * self.training_split))
            self.X_train, self.X_val = self.X_train[:split, :], self.X_train[split:, :]
            self.y_train, self.y_val = self.y_train[:split], self.y_train[split:]


        if (label_noise):
            y_train_noisy = []
            for label in self.y_train:
                if random.random() < label_noise:
                    possible_labels = list(np.unique(y_train))
                    possible_labels.remove(label)
                    # print(label)
                    label = random.choice(possible_labels)
                    # print(label)
                y_train_noisy.append([label])
            self.y_train = y_train_noisy


        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        if (self.training_split < 1.0):
            self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)

        self.set_training_sample(self.X_train.shape[0])


class SPIRAL(Dataset):
    def __init__(self, points, noise, input_shape, random_state=0, training_split=1.0):
        self.num_classes = 2
        np.random.seed(random_state)
        n = np.sqrt(np.random.rand(points, 1)) * 780 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(points, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(points, 1) * noise
        X, y = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
                np.hstack((np.zeros(points), np.ones(points))))
        self.training_split = training_split


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.25, random_state=1)

        self.y_train = np.asarray(self.y_train).astype('float32').reshape((-1, 1))
        self.y_val = np.asarray(self.y_val).astype('float32').reshape((-1, 1))
        self.y_test = np.asarray(self.y_test).astype('float32').reshape((-1, 1))

        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)

        max = np.max(self.X_train)
        self.X_train = self.X_train/np.max(max)
        self.X_val = self.X_val/np.max(max)
        self.X_test = self.X_test/np.max(max)

        self.input_shape = input_shape

        self.set_training_sample(self.X_train.shape[0])


    def visualizeTrainTest(self):
        plt.figure(figsize=(12, 8))
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], color='b', label='class X_train')
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], color='r', label='class X_test')
        plt.xlabel('feature1')
        plt.ylabel('feature2')
        plt.legend()
        plt.axis('equal')
        plt.show()



    def visualizeTest(self):
        plt.figure(figsize=(8, 10))
        plt.subplot(212)
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.prediction_values[:, 0], cmap=cm.coolwarm)
        plt.title('Model predictions on our Test set')
        plt.axis('equal');

    def visualizeDecisionBoundry(self, nn):
        xx = np.linspace(-1, 1, 400)
        yy = np.linspace(-1, 1, 400)
        gx, gy = np.meshgrid(xx, yy)
        Z = nn.model.predict(np.c_[gx.ravel(), gy.ravel()])
        Z = Z.reshape(gx.shape)
        plt.contourf(gx, gy, Z, cmap=cm.coolwarm, alpha=0.8)

        axes = plt.gca()
        axes.set_xlim([-1, 1])
        axes.set_ylim([-1, 1])
        plt.grid('off')
        plt.axis('off')

        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.prediction_values[:, 0], cmap=cm.coolwarm)
        title = 'Neurons: ' + str(nn.neurons) +', Loss: ' + str(np.round(self.testLoss,2)) + ', Accuracy: ' + str(np.round(self.testAccuracy, 2)) + '%'
        plt.title(title)

        #plt.savefig('figure1.pdf')

    def test(self, nn):
        self.prediction_values = (nn.model.predict(self.X_test) > 0.5).astype("int32")
        print("Evaluating on testing set...")
        (loss, accuracy) = nn.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
        self.testAccuracy = 100 * accuracy
        self.testLoss = loss

        (loss, accuracy) = nn.model.evaluate(self.X_val, self.y_val, verbose=0)
        self.valAccuracy = 100 * accuracy
        self.valLoss = loss

class CIRCLE(SPIRAL):
    def __init__(self, points, noise, input_shape, randomState=0, training_split=1.0):
        self.num_classes = 2
        X, y = make_circles(n_samples=points, noise=noise, shuffle=True, random_state=randomState)
        self.training_split = training_split


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.25, random_state=1)

        max = np.max(self.X_train)
        self.X_train = self.X_train/np.max(max)
        self.X_val = self.X_val/np.max(max)
        self.X_test = self.X_test/np.max(max)

        self.y_train = np.asarray(self.y_train).astype('float32').reshape((-1, 1))
        self.y_val = np.asarray(self.y_val).astype('float32').reshape((-1, 1))
        self.y_test = np.asarray(self.y_test).astype('float32').reshape((-1, 1))

        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)


        self.input_shape = input_shape

        self.set_training_sample(self.X_train.shape[0])


class MOON(SPIRAL):
    def __init__(self, points, noise, input_shape, randomState=0, training_split=1.0):
        self.num_classes = 2
        X, y = make_moons(n_samples=points, noise=noise, shuffle=True, random_state=randomState)
        self.training_split = training_split


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.25, random_state=1)



        max = np.max(self.X_train)
        self.X_train = self.X_train/np.max(max)
        self.X_val = self.X_val/np.max(max)
        self.X_test = self.X_test/np.max(max)

        self.y_train = np.asarray(self.y_train).astype('float32').reshape((-1, 1))
        self.y_val = np.asarray(self.y_val).astype('float32').reshape((-1, 1))
        self.y_test = np.asarray(self.y_test).astype('float32').reshape((-1, 1))

        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)

        self.input_shape = input_shape

        self.set_training_sample(self.X_train.shape[0])


