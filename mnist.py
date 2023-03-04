import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def train_model(model, mnist_train, mnist_test, name_model='../model.h5'):
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(mnist_train, mnist_test)
    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=name_model, monitor='val_loss', verbose=2, save_best_only=True)
    model.fit(x_train, y_train, batch_size=150, callbacks=[ckpt], epochs=40, verbose=2, validation_data=(x_val, y_val))


def create_model(input_shape, output_shape):

    inp = tf.keras.layers.Input(shape=(input_shape, ), name='input')
    hdn1 = tf.keras.layers.Dense(units=256, activation='relu', use_bias=True)(inp)
    hdn2 = tf.keras.layers.Dense(units=64, activation='relu', use_bias=True)(hdn1)
    out = tf.keras.layers.Dense(units=output_shape, activation='softmax')(hdn2)

    model = tf.keras.models.Model(inputs=[inp], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics="accuracy")

    return model

def load_data(ds_train, ds_test):
    x_train = np.asanyarray(list(map(lambda x: x[0], ds_train)))
    y_train = np.asanyarray(list(map(lambda x: x[1], ds_train)))
    x_val = np.asanyarray(list(map(lambda x: x[0], ds_test)))
    y_val = np.asanyarray(list(map(lambda x: x[1], ds_test)))

    x_test = x_val[-1000:]
    y_test = y_val[-1000:]
    x_val = x_val[:-1000]
    y_val = y_val[:-1000]

    x_train = np.reshape(x_train, newshape=(x_train.shape[0], -1))
    x_val = np.reshape(x_val, newshape=(x_val.shape[0], -1))
    x_test = np.reshape(x_test, newshape=(x_test.shape[0], -1))

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test

def load_dataset(name='mnist', shuffle_files=True, as_supervised=True, with_info=True):
    mnist_dataset, mnist_info = tfds.load(name=name,
                                          shuffle_files=shuffle_files,
                                          as_supervised=as_supervised,
                                          with_info=with_info)

    mnist_dataset = tfds.as_numpy(mnist_dataset)
    mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

    return mnist_train, mnist_test


if __name__ == '__main__':
    mnist_train, mnist_test = load_dataset()

    # 784 = 28 * 28
    model = create_model(input_shape=784, output_shape=10)
    train_model(model, mnist_train, mnist_test)

    _, _, _, _, x_test, y_test = load_data(mnist_train, mnist_test)
    best_model = tf.keras.models.load_model('../model.h5')
    y_pred = np.argmax(best_model.predict(x_test), axis=-1)

    # for i in range(y_test.shape[0]):
    #     print(y_pred[i], np.argmax(y_test, axis=-1)[i]) # y hat

    cm = np.zeros((10, 10))
    y_test_max = np.argmax(y_test, axis=-1)
    print(y_test_max.shape)
    cnt = 0
    for i in range(y_test_max.shape[0]):
        cm[y_test_max[i], y_pred[i]] += 1
        if y_test_max[i] == y_pred[i]:
            cnt += 1

    print(cm)

    loss, metric = best_model.evaluate(x_test, y_test)
    print(loss, metric)
    print('cate corecte', cnt, cnt / y_test_max.shape[0])
