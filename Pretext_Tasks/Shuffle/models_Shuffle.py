from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input, Conv3D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import Sequential


#Modelo para la tarea de pretexto de reconocer frames desordenados
def model_Shuffle_CONV3D(the_input_shape, dropout_rate_1, dropout_rate_2, units_dense_layer, learning_rate):

    #Se define la entrada del modelo
    inputs = Input(the_input_shape)

    base_model = Sequential(name='CONV3D')

    base_model.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', input_shape=the_input_shape, name='Conv3D_1_CONV3D'))

    base_model.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_2_CONV3D'))

    base_model.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_3_CONV3D'))

    base_model.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_4_CONV3D'))


    output_1 = base_model(inputs)

    #Se definen las capas de clasificación del modelo
    x = Dropout(dropout_rate_1, name='Dropout_1_Shuffle')(output_1)

    features = Flatten(name='Flatten_Shuffle')(x)

    x = Dense(units=units_dense_layer, activation='relu', name='FC_1_Shuffle')(features)

    x = Dropout(dropout_rate_2, name='Dropout_2_Shuffle')(x)

    outputs = Dense(2, activation='softmax', name='FC_Final_Shuffle')(x)

    #Se define el modelo
    model = Model(inputs, outputs)

    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

"""def model_Shuffle_C3D(the_input_shape, dropout_rate_1, dropout_rate_2, units_dense_layers_1, units_dense_layers_2, learning_rate):

    #Se define la entrada del modelo
    inputs = Input(the_input_shape)

    # Se declara el modelo base que se va a emplear (capas convoluciones del modelo)
    basemodel = C3D(the_input_shape)

    x = basemodel(inputs, training=True)

    features = Flatten(name='Flatten_Shuffle')(x)

    x = Dense(units=units_dense_layers_1, activation='relu', name='FC_1_Shuffle')(features)

    x = Dropout(dropout_rate_1, name='Dropout_1_Shuffle')(x)

    x = Dense(units=units_dense_layers_2, activation='relu', name='FC_2_Shuffle')(x)

    x = Dropout(dropout_rate_2, name='Dropout_2_Shuffle')(x)

    outputs = Dense(2, activation='softmax', name='FC_Final_Shuffle')(x)

    #Se define el modelo
    model = Model(inputs, outputs)

    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model"""