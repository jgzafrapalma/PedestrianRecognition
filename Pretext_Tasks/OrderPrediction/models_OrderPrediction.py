from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam

import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.append(os.path.join(rootdir, 'base_models'))

from CaffeNet import CaffeNet

def model_OrderPrediction_SIAMESE(the_input_shape, units_dense_layers_1, units_dense_layers_2, learning_rate):
    # Se definen las 4 entradas del modelo
    input_1 = Input(shape=the_input_shape)
    input_2 = Input(shape=the_input_shape)
    input_3 = Input(shape=the_input_shape)
    input_4 = Input(shape=the_input_shape)

    base_model = CaffeNet(the_input_shape)

    # Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
    output_1 = base_model(input_1)
    output_2 = base_model(input_2)
    output_3 = base_model(input_3)
    output_4 = base_model(input_4)

    flatten = Flatten(name='Flatten_OrderPrediction')

    # Se obtienen los vectores de características de las 4 entradas
    features_1 = flatten(output_1)
    features_2 = flatten(output_2)
    features_3 = flatten(output_3)
    features_4 = flatten(output_4)

    # Capa densa utilizada para resumir las caracteristicas extraidas de las capas convolucionales para cada frame
    dense_1 = Dense(units=units_dense_layers_1, activation='relu', name='FC_1_OrderPrediction')

    features_1 = dense_1(features_1)
    features_2 = dense_1(features_2)
    features_3 = dense_1(features_3)
    features_4 = dense_1(features_4)

    Features_12 = concatenate([features_1, features_2])
    Features_13 = concatenate([features_1, features_3])
    Features_14 = concatenate([features_1, features_4])
    Features_23 = concatenate([features_2, features_3])
    Features_24 = concatenate([features_2, features_4])
    Features_34 = concatenate([features_3, features_4])

    # Capa densa que aprende la relación entre las características de los distintos fotogramas
    dense_2 = Dense(units=units_dense_layers_2, activation='relu', name='FC_2_OrderPrediction')

    RelationShip_1_2 = dense_2(Features_12)
    RelationShip_1_3 = dense_2(Features_13)
    RelationShip_1_4 = dense_2(Features_14)
    RelationShip_2_3 = dense_2(Features_23)
    RelationShip_2_4 = dense_2(Features_24)
    RelationShip_3_4 = dense_2(Features_34)

    # Concatenación de todas las relaciones
    Features_Final = concatenate(
        [RelationShip_1_2, RelationShip_1_3, RelationShip_1_4, RelationShip_2_3, RelationShip_2_4, RelationShip_3_4])

    prediction = Dense(units=12, activation='softmax', name='FC_Final_OrderPrediction')(Features_Final)

    siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

    siamese_model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    siamese_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return siamese_model