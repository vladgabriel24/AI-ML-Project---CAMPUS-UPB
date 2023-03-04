import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.python
import tensorflow.python.keras.layers
import os

def create_model(input_shape, output_shape):

    inp = tf.keras.layers.Input(shape=(32,32,3), name='input')

    hdn = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),  activation='relu')(inp)
    hdn = tf.keras.layers.BatchNormalization()(hdn)
    hdn = tf.keras.layers.Dropout(0.1)(hdn)

    hdn = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),  activation='relu')(hdn)
    hdn = tf.keras.layers.BatchNormalization()(hdn)
    hdn = tf.keras.layers.Dropout(0.1)(hdn)

    hdn = tf.keras.layers.MaxPool2D(pool_size=(2,2))(hdn)

    hdn = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),  activation='relu')(hdn)
    hdn = tf.keras.layers.BatchNormalization()(hdn)
    hdn = tf.keras.layers.Dropout(0.1)(hdn)

    hdn = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),  activation='relu')(hdn)
    hdn = tf.keras.layers.BatchNormalization()(hdn)

    hdn = tf.keras.layers.MaxPool2D(pool_size=(2,2))(hdn)
    hdn = tf.keras.layers.Dropout(0.4)(hdn)

    hdn = tf.keras.layers.Flatten()(hdn)
    hdn = tf.keras.layers.Dropout(0.3)(hdn)

    hdn = tf.keras.layers.Dense(2048, activation='relu')(hdn)
    hdn = tf.keras.layers.Dropout(0.3)(hdn)

    hdn = tf.keras.layers.Dense(512, activation='relu')(hdn)
    hdn = tf.keras.layers.Dropout(0.4)(hdn)

    out = tf.keras.layers.Dense(10, activation='softmax')(hdn)

    model1 = tf.keras.models.Model(inputs = [inp], outputs = [out])

    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics="accuracy")

    model1.summary()

    return model1


def train_model(model, x_train, x_test, y_train, y_test,  name_model=''):

    # x_train = np.asanyarray(list(map(lambda x : x.flatten(), x_train)))
    # x_test = np.asanyarray(list(map(lambda x : x.flatten(), x_test)))

    # y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    # y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=name_model + "/" + "model_{val_accuracy:.4f}.h5" , monitor='val_loss', verbose=1, save_best_only=True)
    history = model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(x_test, y_test), callbacks = [ckpt])


(x_train, y_train) , (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalizare <=> Feature scaling : realizata pentru a functiona corect gradient descent-ul
x_train = x_train/255.0
x_test = x_test/255.0

# One hot encoding
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)


model = create_model(3072, 10)
name_model = 'model1'
if not os.path.exists(name_model):
    os.mkdir(name_model)
    
# train_model(model, x_train, x_test, y_train, y_test, name_model)

best_model = tf.keras.models.load_model('model_0.8109.h5')

# x_test_copy = np.asanyarray(list(map(lambda x : x.flatten(), x_test)))
# y_test_copy = tf.keras.utils.to_categorical(y_test, num_classes=10)

loss, metric = best_model.evaluate(x_test, y_test)

print(loss, metric)



