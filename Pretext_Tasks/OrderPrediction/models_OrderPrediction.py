from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam

import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.append(os.path.join(rootdir, 'base_models'))

from CaffeNet import CaffeNet

def model_OrderPrediction_SIAMESE(the_input_shape, learning_rate):
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

    flatten_1 = Flatten(name='Flatten_1_OrderPrediction')

    # Se obtienen los vectores de características de las 4 entradas
    features_1 = flatten_1(output_1)
    features_2 = flatten_1(output_2)
    features_3 = flatten_1(output_3)
    features_4 = flatten_1(output_4)

    # Se añade una capa customizada que permite realizar la contatenación de dos vectores de características

    # Concatenate_Features = Lambda(lambda tensors: keras.backend.concatenate((tensors[0], tensors[1])))

    Features_12 = concatenate([features_1, features_2])
    Features_13 = concatenate([features_1, features_3])
    Features_14 = concatenate([features_1, features_4])
    Features_23 = concatenate([features_2, features_3])
    Features_24 = concatenate([features_2, features_4])
    Features_34 = concatenate([features_3, features_4])

    dense1 = Dense(units=512, activation='relu', name='FC_1_OrderPrediction')

    RelationShip_12 = dense1(Features_12)
    RelationShip_13 = dense1(Features_13)
    RelationShip_14 = dense1(Features_14)
    RelationShip_23 = dense1(Features_23)
    RelationShip_24 = dense1(Features_24)
    RelationShip_34 = dense1(Features_34)

    # Concatenate_RelationShips = Lambda(lambda tensors: keras.backend.concatenate((tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5])))

    Features_Final = concatenate(
        [RelationShip_12, RelationShip_13, RelationShip_14, RelationShip_23, RelationShip_24, RelationShip_34])

    prediction = Dense(units=12, activation='softmax', name='FC_Final_OrderPrediction')(Features_Final)

    siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

    siamese_model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    siamese_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return siamese_model