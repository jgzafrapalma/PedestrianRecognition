from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, concatenate, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras import Sequential

import tensorflow as tf

import numpy as np

def model_CrossingDetection_OrderPrediction_SIAMESE_TL(the_input_shape, units_dense_layer_1, units_dense_layer_2, dropout_rate_1, dropout_rate_2, learning_rate, path_conv_weights):

    # Se definen las 4 entradas del modelo
    input_1 = Input(shape=the_input_shape)
    input_2 = Input(shape=the_input_shape)
    input_3 = Input(shape=the_input_shape)
    input_4 = Input(shape=the_input_shape)

    
    #CaffeNet
    base_model = Sequential()

    base_model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', data_format='channels_last',
                    activation='relu', input_shape=the_input_shape, name='Conv2D_1_CaffeNet'))
    base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_1_CaffeNet'))
    base_model.add(BatchNormalization())
    
    base_model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_2_CaffeNet'))
    base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_2_CaffeNet'))
    base_model.add(BatchNormalization())

    base_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_3_CaffeNet'))
    
    base_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_4_CaffeNet'))

    base_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_5_CaffeNet'))

    base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_3_CaffeNet'))

    
    #Se congelan las capas del modelo base
    base_model.trainable = False

    #Se cargan los pesos en las capas convoluciones para la transferencia de conocimiento
    with (path_conv_weights).open('rb') as file_descriptor:
        conv_weights = np.load(file_descriptor, allow_pickle=True)

    base_model.set_weights(conv_weights)

    # Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
    output_1 = base_model(input_1, training=False)
    output_2 = base_model(input_2, training=False)
    output_3 = base_model(input_3, training=False)
    output_4 = base_model(input_4, training=False)

    flatten_1 = Flatten(name='Flatten_1_FINAL')

    # Se obtienen los vectores de características de las 4 entradas
    features_1 = flatten_1(output_1)
    features_2 = flatten_1(output_2)
    features_3 = flatten_1(output_3)
    features_4 = flatten_1(output_4)

    # Capa densa utilizada para resumir las caracteristicas extraidas de las capas convolucionales para cada frame
    dense_1 = Dense(
        units=units_dense_layer_1,
        activation='relu',
        name='FC_1_FINAL'
    )

    features_1 = dense_1(features_1)
    features_2 = dense_1(features_2)
    features_3 = dense_1(features_3)
    features_4 = dense_1(features_4)

    dropout_1 = Dropout(
        rate=dropout_rate_1,
        name='Dropout_1_FINAL'
    )

    features_1 = dropout_1(features_1)
    features_2 = dropout_1(features_2)
    features_3 = dropout_1(features_3)
    features_4 = dropout_1(features_4)

    Features_12 = concatenate([features_1, features_2])
    Features_13 = concatenate([features_1, features_3])
    Features_14 = concatenate([features_1, features_4])
    Features_23 = concatenate([features_2, features_3])
    Features_24 = concatenate([features_2, features_4])
    Features_34 = concatenate([features_3, features_4])

    # Capa densa que aprende la relación entre las características de los distintos fotogramas
    dense_2 = Dense(
        units=units_dense_layer_2,
        activation='relu',
        name='FC_2_FINAL'
    )

    RelationShip_1_2 = dense_2(Features_12)
    RelationShip_1_3 = dense_2(Features_13)
    RelationShip_1_4 = dense_2(Features_14)
    RelationShip_2_3 = dense_2(Features_23)
    RelationShip_2_4 = dense_2(Features_24)
    RelationShip_3_4 = dense_2(Features_34)

    dropout_2 = Dropout(
        rate=dropout_rate_2,
        name='Dropout_2_FINAL'
    )

    RelationShip_1_2 = dropout_2(RelationShip_1_2)
    RelationShip_1_3 = dropout_2(RelationShip_1_3)
    RelationShip_1_4 = dropout_2(RelationShip_1_4)
    RelationShip_2_3 = dropout_2(RelationShip_2_3)
    RelationShip_2_4 = dropout_2(RelationShip_2_4)
    RelationShip_3_4 = dropout_2(RelationShip_3_4)

    # Concatenación de todas las relaciones
    Features_Final = concatenate(
        [RelationShip_1_2, RelationShip_1_3, RelationShip_1_4, RelationShip_2_3, RelationShip_2_4, RelationShip_3_4])

    prediction = Dense(units=2, activation='softmax', name='FC_Final_FINAL')(Features_Final)

    siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

    siamese_model.summary()

    optimizer = SGD(learning_rate=learning_rate)

    siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return siamese_model



def model_CrossingDetection_OrderPrediction_SIAMESE_NTL(the_input_shape, units_dense_layer_1, units_dense_layer_2, dropout_rate_1, dropout_rate_2, learning_rate):

    # Se definen las 4 entradas del modelo
    input_1 = Input(shape=the_input_shape)
    input_2 = Input(shape=the_input_shape)
    input_3 = Input(shape=the_input_shape)
    input_4 = Input(shape=the_input_shape)

    
    #CaffeNet
    base_model = Sequential()

    base_model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', data_format='channels_last',
                    activation='relu', input_shape=the_input_shape, name='Conv2D_1_CaffeNet'))
    base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_1_CaffeNet'))
    base_model.add(BatchNormalization())
    
    base_model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_2_CaffeNet'))
    base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_2_CaffeNet'))
    base_model.add(BatchNormalization())

    base_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_3_CaffeNet'))
    
    base_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_4_CaffeNet'))

    base_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_5_CaffeNet'))

    base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_3_CaffeNet'))

    
    # Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
    output_1 = base_model(input_1)
    output_2 = base_model(input_2)
    output_3 = base_model(input_3)
    output_4 = base_model(input_4)

    flatten_1 = Flatten(name='Flatten_1_FINAL')

    # Se obtienen los vectores de características de las 4 entradas
    features_1 = flatten_1(output_1)
    features_2 = flatten_1(output_2)
    features_3 = flatten_1(output_3)
    features_4 = flatten_1(output_4)

    # Capa densa utilizada para resumir las caracteristicas extraidas de las capas convolucionales para cada frame
    dense_1 = Dense(
        units=units_dense_layer_1,
        activation='relu',
        name='FC_1_FINAL'
    )

    features_1 = dense_1(features_1)
    features_2 = dense_1(features_2)
    features_3 = dense_1(features_3)
    features_4 = dense_1(features_4)

    dropout_1 = Dropout(
        rate=dropout_rate_1,
        name='Dropout_1_FINAL'
    )

    features_1 = dropout_1(features_1)
    features_2 = dropout_1(features_2)
    features_3 = dropout_1(features_3)
    features_4 = dropout_1(features_4)

    Features_12 = concatenate([features_1, features_2])
    Features_13 = concatenate([features_1, features_3])
    Features_14 = concatenate([features_1, features_4])
    Features_23 = concatenate([features_2, features_3])
    Features_24 = concatenate([features_2, features_4])
    Features_34 = concatenate([features_3, features_4])

    # Capa densa que aprende la relación entre las características de los distintos fotogramas
    dense_2 = Dense(
        units=units_dense_layer_2,
        activation='relu',
        name='FC_2_FINAL'
    )

    RelationShip_1_2 = dense_2(Features_12)
    RelationShip_1_3 = dense_2(Features_13)
    RelationShip_1_4 = dense_2(Features_14)
    RelationShip_2_3 = dense_2(Features_23)
    RelationShip_2_4 = dense_2(Features_24)
    RelationShip_3_4 = dense_2(Features_34)

    dropout_2 = Dropout(
        rate=dropout_rate_2,
        name='Dropout_2_FINAL'
    )

    RelationShip_1_2 = dropout_2(RelationShip_1_2)
    RelationShip_1_3 = dropout_2(RelationShip_1_3)
    RelationShip_1_4 = dropout_2(RelationShip_1_4)
    RelationShip_2_3 = dropout_2(RelationShip_2_3)
    RelationShip_2_4 = dropout_2(RelationShip_2_4)
    RelationShip_3_4 = dropout_2(RelationShip_3_4)

    # Concatenación de todas las relaciones
    Features_Final = concatenate(
        [RelationShip_1_2, RelationShip_1_3, RelationShip_1_4, RelationShip_2_3, RelationShip_2_4, RelationShip_3_4])

    prediction = Dense(units=2, activation='softmax', name='FC_Final_FINAL')(Features_Final)

    siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

    siamese_model.summary()

    optimizer = SGD(learning_rate=learning_rate)

    siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return siamese_model