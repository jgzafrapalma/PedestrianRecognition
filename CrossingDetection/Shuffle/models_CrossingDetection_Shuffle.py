from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input, Conv3D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import Sequential

import tensorflow as tf

import numpy as np


def model_CrossingDetection_Shuffle_CONV3D_TL(the_input_shape, dropout_rate_1, dropout_rate_2, units_dense_layer, learning_rate, path_conv_weights):


    # Se define la entrada del modelo
    inputs = Input(the_input_shape)

    base_model = Sequential(name='CONV3D')

    base_model.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', input_shape=the_input_shape, name='Conv3D_1_CONV3D'))

    base_model.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_2_CONV3D'))

    base_model.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_3_CONV3D'))

    base_model.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_4_CONV3D'))

    # Se congela el modelo base para que sus pesos no sean entrenables
    base_model.trainable = False

    #Se cargan los pesos en las capas convoluciones para la transferencia de conocimiento
    with (path_conv_weights).open('rb') as file_descriptor:
        conv_weights = np.load(file_descriptor, allow_pickle=True)

    base_model.set_weights(conv_weights)

    output_1 = base_model(inputs, training=False)

    x = Dropout(dropout_rate_1, name='Dropout_1_FINAL')(output_1)

    x = Flatten(name='Flatten_FINAL')(x)

    x = Dense(units=units_dense_layer, activation='relu', name='FC_1_FINAL')(x)

    x = Dropout(dropout_rate_2, name='Dropout_2_FINAL')(x)

    outputs = Dense(2, activation='softmax', name='FC_2_FINAL')(x)

    # Se define el modelo
    model = Model(inputs, outputs)

    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

def model_CrossingDetection_Shuffle_CONV3D_NTL(the_input_shape, dropout_rate_1, dropout_rate_2, units_dense_layer, learning_rate):


    # Se define la entrada del modelo
    inputs = Input(the_input_shape)

    base_model = Sequential(name='CONV3D')

    base_model.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', input_shape=the_input_shape, name='Conv3D_1_CONV3D'))

    base_model.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_2_CONV3D'))

    base_model.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_3_CONV3D'))

    base_model.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_4_CONV3D'))


    output_1 = base_model(inputs)

    x = Dropout(dropout_rate_1, name='Dropout_1_FINAL')(output_1)

    x = Flatten(name='Flatten_FINAL')(x)

    x = Dense(units=units_dense_layer, activation='relu', name='FC_1_FINAL')(x)

    x = Dropout(dropout_rate_2, name='Dropout_2_FINAL')(x)

    outputs = Dense(2, activation='softmax', name='FC_2_FINAL')(x)

    # Se define el modelo
    model = Model(inputs, outputs)

    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model
