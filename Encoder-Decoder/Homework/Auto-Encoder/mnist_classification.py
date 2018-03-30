import utils
from keras.models import *
from keras.layers import Dense, Dropout

def y2tag(y):
    tag = np.zeros(shape=(y.shape[0], 10))
    for i in range(y.shape[0]):
        tag[i, y[i]] = 1
    return tag

def predict2y(predict):
    return np.argmax(predict, axis=1)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = utils.load_data()
    x_train = np.reshape(x_train, newshape=(x_train.shape[0], -1))      ## shape=(60000, 28, 28) >>> (60000, 784)
    x_train = x_train / 255     ## normalization
    y_tag = y2tag(y_train)
    x_test = np.reshape(x_test, newshape=(x_test.shape[0], -1))  / 255

    ## model
    x = Input(shape=(784,))
    layer = Dense(32, activation='relu')(x)
    layer = Dropout(rate=0.2)(layer)
    layer = Dense(32, activation='relu')(layer)
    layer = Dropout(rate=0.2)(layer)
    y = Dense(10, activation='softmax')(layer)
    model = Model(inputs=x, outputs=y)
    model.compile('sgd', 'binary_crossentropy')
    ## train
    model.fit(x_train, y_tag, epochs=10)
    ## test
    predict = model.predict(x_test)
    predict = predict2y(predict)
    ## evaluate
    count = 0
    for i in range(y_test.shape[0]):
        if y_test[i] == predict[i]:
            count += 1
    print('accuracy: %.2f%%' % (100 * count / y_test.shape[0]))

