import numpy as np
from keras.models import *
from keras.layers import Dense
import matplotlib.pyplot as plt

def load_data(path='mnist.npz'):
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def encoder(x, activation):
    l1 = Dense(128, activation=activation)(x)
    l2 = Dense(32, activation=activation)(l1)
    l3 = Dense(8, activation=activation)(l2)
    l4 = Dense(4, activation=activation)(l3)
    code = Dense(1, activation=activation)(l4)
    return code

def decoder(code, activation):
    l0 = Dense(4, activation=activation)(code)
    l1 = Dense(8, activation=activation)(l0)
    l2 = Dense(32, activation=activation)(l1)
    l3 = Dense(128, activation=activation)(l2)
    y = Dense(784, activation=activation)(l3)
    return y

def auto_encoder(activation):
    x = Input(shape=(784,))
    code = encoder(x, activation)
    y = decoder(code, activation)
    model = Model(inputs=x, outputs=y)
    model.compile('rmsprop', 'mse')
    return model

def train_auto_encoder(x, epochs, activation):
    x = np.reshape(x, newshape=(x.shape[0], -1))  ## shape=(60000, 28, 28) >>> (60000, 784)
    x = x / 255  ## normalization
    model = auto_encoder(activation)
    model.fit(x, x, epochs=epochs)
    model.save_weights('your_weights_%s' % activation, overwrite=True)
    return model, 'your_weights_%s' % activation

def reconstruct_x(x, model):
    x = np.reshape(x, newshape=(-1, 784)) / 255
    x_r = model.predict(x)
    x_r = np.reshape(x_r, newshape=(28, 28)) * 255
    return x_r

def compare(x, x_l, x_nl):
    p1 = plt.subplot(131)
    p2 = plt.subplot(132)
    p3 = plt.subplot(133)

    p1.imshow(x, cmap='Greys_r')
    p2.imshow(x_l, cmap='Greys_r')
    p3.imshow(x_nl, cmap='Greys_r')
    plt.show()

def get_code(x0, src):
    x0 = np.reshape(x0, newshape=(x0.shape[0], -1))  ## shape=(60000, 28, 28) >>> (60000, 784)
    x0 = x0 / 255  ## normalization
    if 'sigmoid' in src:
        activation = 'sigmoid'
    else:
        activation = None
    x = Input(shape=(784,))
    code = encoder(x, activation)
    y = decoder(code, activation)
    model = Model(inputs=x, outputs=[y, code])
    model.load_weights(src)
    p, c = model.predict(x0)
    return c

def find(array, value):
    result = []
    for i in range(len(array)):
        if array[i] == value:
            result.append(i)
    return result

def plot(x, label):
    for i in range(10):
        idx = find(label, i)
        plt.scatter(x[idx, 0], np.random.standard_normal(size=(len(idx))), label='%d'%i, s=5)
    plt.legend(loc='best')
    plt.show()