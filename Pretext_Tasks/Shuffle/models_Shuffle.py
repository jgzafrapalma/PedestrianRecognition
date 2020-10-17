import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.append(os.path.join(rootdir, 'base_models'))

from base_models import CONV3D, C3D

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam


#Modelo para la tarea de pretexto de reconocer frames desordenados
def model_Shuffle_CONV3D(the_input_shape, dropout_rate_1, dropout_rate_2, dense_activation, units_dense_layer, learning_rate):

    #Se define la entrada del modelo
    inputs = Input(the_input_shape)

    # Se declara el modelo base que se va a emplear (capas convoluciones del modelo)
    basemodel = CONV3D(the_input_shape)

    x = basemodel(inputs, training=True)

    #Se definen las capas de clasificación del modelo
    x = Dropout(dropout_rate_1, name='Dropout_1_Shuffle')(x)

    features = Flatten(name='Flatten_Shuffle')(x)

    x = Dense(units=units_dense_layer, activation=dense_activation, name='FC_1_Shuffle')(features)

    x = Dropout(dropout_rate_2, name='Dropout_2_Shuffle')(x)

    outputs = Dense(2, activation='softmax', name='FC_Final_Shuffle')(x)

    #Se define el modelo
    model = Model(inputs, outputs)

    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def model_Shuffle_C3D(the_input_shape, dropout_rate_1, dropout_rate_2, dense_activation, units_dense_layers_1, units_dense_layers_2, learning_rate):

    #Se define la entrada del modelo
    inputs = Input(the_input_shape)

    # Se declara el modelo base que se va a emplear (capas convoluciones del modelo)
    basemodel = C3D(the_input_shape)

    x = basemodel(inputs, training=True)

    features = Flatten(name='Flatten_Shuffle')(x)

    x = Dense(units=units_dense_layers_1, activation=dense_activation, name='FC_1_Shuffle')(features)

    x = Dropout(dropout_rate_1, name='Dropout_1_Shuffle')(x)

    x = Dense(units=units_dense_layers_2, activation=dense_activation, name='FC_2_Shuffle')(x)

    x = Dropout(dropout_rate_2, name='Dropout_2_Shuffle')(x)

    outputs = Dense(2, activation='softmax', name='FC_Final_Shuffle')(x)

    #Se define el modelo
    model = Model(inputs, outputs)

    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model