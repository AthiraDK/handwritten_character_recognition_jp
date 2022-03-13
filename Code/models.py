import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import BatchNormalization
from tensorflow.keras import activations

def M11(n_classes, input_shape=(1, 64, 64)):

    model = keras.Sequential()
    model.add(layers.Conv2D(64, (3,3),  padding='same', kernel_initializer='he_uniform', name='conv3_64_1',
                                  input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', name='conv3_64_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2,2), name='pool_64'))

    model.add(layers.Conv2D(128, (3,3), padding='same',  kernel_initializer='he_uniform', name='conv3_128_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(128, (3,3),padding='same',  kernel_initializer='he_uniform', name='conv3_128_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_128'))

    model.add(layers.Conv2D(256, (3,3), padding='same',  kernel_initializer='he_uniform', name='conv3_256_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_uniform', name='conv3_256_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_256'))

    model.add(layers.Conv2D(512, (3,3), padding='same',  kernel_initializer='he_uniform', name='conv3_512_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_uniform', name='conv3_512_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_512'))

    model.add(layers.Flatten())

    model.add(layers.Dense(1024,kernel_initializer='he_uniform', name='fc_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1024,kernel_initializer='he_uniform', name='fc_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(n_classes,activation='softmax',name='fc_final'))
    return model

def M12(n_classes, input_shape=(1, 64, 64)):

    model = keras.Sequential()
    model.add(layers.Conv2D(64, (3,3),  padding='same', kernel_initializer='he_uniform', name='conv3_64_1',
                                  input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', name='conv3_64_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2,2), name='pool_64'))

    model.add(layers.Conv2D(128, (3,3), padding='same',  kernel_initializer='he_uniform', name='conv3_128_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(128, (3,3),padding='same',  kernel_initializer='he_uniform', name='conv3_128_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_128'))

    model.add(layers.Conv2D(256, (3,3), padding='same',  kernel_initializer='he_uniform', name='conv3_256_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_uniform', name='conv3_256_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_256'))

    model.add(layers.Conv2D(512, (3,3), padding='same',  kernel_initializer='he_uniform', name='conv3_512_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_uniform', name='conv3_512_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_uniform', name='conv3_512_3'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_512'))

    model.add(layers.Flatten())

    model.add(layers.Dense(4096,kernel_initializer='he_uniform', name='fc_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4096,kernel_initializer='he_uniform', name='fc_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(n_classes,activation='softmax',name='fc_final'))
    return model

def M7_1(n_classes, input_shape=(1, 64, 64)):

    model = keras.Sequential()
    model.add(layers.Conv2D(64, (3,3),  padding='same', kernel_initializer='he_uniform', name='conv3_64',
                                  input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2,2), name='pool_64'))

    model.add(layers.Conv2D(128, (3,3), padding='same',  kernel_initializer='he_uniform', name='conv3_128'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_128'))

    model.add(layers.Conv2D(512, (3,3), padding='same',  kernel_initializer='he_uniform', name='conv3_512_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_uniform', name='conv3_512_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_512'))

    model.add(layers.Flatten())

    model.add(layers.Dense(4096,kernel_initializer='he_uniform', name='fc_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4096,kernel_initializer='he_uniform', name='fc_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(n_classes,activation='softmax',name='fc_final'))
    return model

def M7_2(n_classes, input_shape=(1, 64, 64)):

    model = keras.Sequential()
    model.add(layers.Conv2D(64, (3,3),  padding='same', kernel_initializer='he_uniform', name='conv3_64',
                                  input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2,2), name='pool_64'))

    model.add(layers.Conv2D(128, (3,3), padding='same',  kernel_initializer='he_uniform', name='conv3_128'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_128'))

    model.add(layers.Conv2D(192, (3,3), padding='same',  kernel_initializer='he_uniform', name='conv3_192'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_192'))

    model.add(layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_uniform', name='conv3_256'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2), name='pool_256'))

    model.add(layers.Flatten())

    model.add(layers.Dense(1024,kernel_initializer='he_uniform', name='fc_1'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1024,kernel_initializer='he_uniform', name='fc_2'))
    model.add(BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(n_classes,activation='softmax',name='fc_final'))
    return model
