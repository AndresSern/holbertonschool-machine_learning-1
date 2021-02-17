#!/usr/bin/env python3
"""trains a convolutional neural network to classify the CIFAR 10 dataset"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """pre-processes the data"""
    X_p = X_p = K.applications.densenet.preprocess_input(X)
    """one hot encode target values"""
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    """  convolutional neural network to classify the CIFAR 10 dataset:"""
    optimizer = K.optimizers.Adam()
    kernel_init = K.initializers.he_normal()

    CALLBACKS = []
    """load dataset"""
    (trainX, trainy), (testX, testy) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(trainX, trainy)
    x_test, y_test = preprocess_data(testX, testy)

    input_x = K.Input(shape=(224, 224, 3))

    """ USE DenseNet121"""
    OldModel = K.applications.DenseNet121(include_top=False,
                                          input_tensor=input_x,
                                          weights='imagenet')

    for layer in OldModel.layers[:149]:
        layer.trainable = False
    for layer in OldModel.layers[149:]:
        layer.trainable = True

    model = K.models.Sequential()
    """a lambda layer that scales up the data to the correct size"""
    model.add(K.layers.Lambda(lambda x:
                              K.backend.resize_images
                              (x, height_factor=7,
                               width_factor=7,
                               data_format='channels_last')))
    """ or model.add(K.layers.Lambda(lambda x: tf.image.resize(x,(32,32))))"""
    model.add(OldModel)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation='relu',
                             kernel_initializer=kernel_init,
                             kernel_regularizer=K.regularizers.l2(0.001)))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu',
                             kernel_initializer=kernel_init,
                             kernel_regularizer=K.regularizers.l2(0.001)))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu',
                             kernel_initializer=kernel_init,
                             kernel_regularizer=K.regularizers.l2(0.001)))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.Dense(10, activation='softmax',
                             kernel_initializer=kernel_init,
                             kernel_regularizer=K.regularizers.l2(0.001)))
    """callbacks"""
    CALLBACKS.append(K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                                 monitor='val_acc',
                                                 save_best_only=True))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    """train"""
    model.fit(x=x_train,
              y=y_train,
              batch_size=128,
              epochs=20,
              callbacks=CALLBACKS,
              validation_data=(x_test, y_test))
    model.summary()
    model.save('cifar10.h5')
