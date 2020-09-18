from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#Modelo para la tarea de pretexto de reconocer frames desordenados
def model_Shuffle_CONV3D(the_input_shape, dropout_rate_1, dropout_rate_2, dense_activation, units_dense_layer, learning_rate):

    model = Sequential()

    model.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu', input_shape=the_input_shape, name='conv3d_1_shuffle'))

    model.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu', name='conv3d_2_shuffle'))

    model.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu', name='conv3d_3_shuffle'))

    model.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu', name='conv3d_1_shuffle'))

    model.add(Dropout(dropout_rate_1, name='dropout_1_shuffle'))

    model.add(Flatten(), name='flatten_shuffle')

    model.add(Dense(units=units_dense_layer, activation=dense_activation, name='fc_1_shuffle'))

    model.add(Dropout(dropout_rate_2, name='dropout_2_shuffle'))

    model.add(Dense(2, activation='softmax', name='fc_2_shuffle'))

    #model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

#Modelo final de regresión cuya tarea de pretexto es Shuffle utilizando un modelo de convolución 3D
def model_FINAL_Shuffle_CONV3D_CrossingDetection(the_input_shape, dropout_rate_1, dropout_rate_2, dense_activation, units_dense_layer, learning_rate):

    model = Sequential()

    conv3d_1_shuffle = Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                        activation='relu', input_shape=the_input_shape, name='conv3d_1_shuffle')
    
    conv3d_1_shuffle.trainable = False

    model.add(conv3d_1_shuffle)

    conv3d_2_shuffle = Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                        activation='relu', name='conv3d_2_shuffle')

    conv3d_2_shuffle.trainable = False

    model.add(conv3d_2_shuffle)

    conv3d_3_shuffle = Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                        activation='relu', name='conv3d_3_shuffle')

    conv3d_3_shuffle.trainable = False

    model.add(conv3d_3_shuffle)

    conv3d_4_shuffle = Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                        activation='relu', name='conv3d_4_shuffle')

    conv3d_4_shuffle.trainable = False

    model.add(conv3d_4_shuffle)

    model.add(Dropout(dropout_rate_1, name='dropout_1_final'))

    model.add(Flatten(name='flatten_final'))

    model.add(Dense(units=units_dense_layer, activation=dense_activation, name='fc_1_final'))

    model.add(Dropout(dropout_rate_2, name='dropout_2_final'))

    model.add(Dense(2, activation='softmax', name='fc_2_final'))

    #model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model