import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
rootdir = os.path.dirname(parentparentdir)
sys.path.append(os.path.join(rootdir, 'base_models'))

from base_models import CONV3D

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam



def model_CrossingDetection_Shuffle_CONV3D(the_input_shape, dropout_rate_1, dropout_rate_2, dense_activation, units_dense_layer, learning_rate):


    # Se define la entrada del modelo
    inputs = Input(the_input_shape)

    # Se declara el modelo base que se va a emplear (capas convoluciones del modelo)
    basemodel = CONV3D(the_input_shape)

    # Se congela el modelo base para que sus pesos no sean entrenables
    basemodel.trainable = False

    # El modelo base se pone en modo inferencia
    x = basemodel(inputs, training=False)

    x = Dropout(dropout_rate_1, name='Dropout_1_FINAL')(x)

    x = Flatten(name='Flatten_FINAL')(x)

    x = Dense(units=units_dense_layer, activation=dense_activation, name='FC_1_FINAL')(x)

    x = Dropout(dropout_rate_2, name='Dropout_2_FINAL')(x)

    outputs = Dense(2, activation='softmax', name='FC_2_FINAL')(x)

    # Se define el modelo
    model = Model(inputs, outputs)

    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model