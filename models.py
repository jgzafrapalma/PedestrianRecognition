from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, Flatten, Dropout, Dense, Conv2D, MaxPooling2D, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras


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

    model.summary()

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


def model_OrderPrediction_SIAMESE(the_input_shape, learning_rate):

    #Se definen las 4 entradas del modelo
    input_1 = Input(shape=the_input_shape)
    input_2 = Input(shape=the_input_shape)
    input_3 = Input(shape=the_input_shape)
    input_4 = Input(shape=the_input_shape)

    model = Sequential()

    #1th Convolutional Layer
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', data_format='channels_last',
                     activation='relu', input_shape=the_input_shape, name='conv2d_1_OrderPrediction'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))
    model.add(BatchNormalization())

    #2th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last',
                     activation='relu', name='conv2d_2_OrderPrediction'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))
    model.add(BatchNormalization())

    #3th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                     activation='relu', name='conv2d_3_OrderPrediction'))

    #4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                     activation='relu', name='conv2d_4_OrderPrediction'))

    #5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                     activation='relu', name='conv2d_5_OrderPrediction'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))

    model.add(Flatten())

    #Se obtienen los vectores de características de las 4 entradas
    features_1 = model(input_1)
    features_2 = model(input_2)
    features_3 = model(input_3)
    features_4 = model(input_4)

    #Se añade una capa customizada que permite realizar la contatenación de dos vectores de características

    #Concatenate_Features = Lambda(lambda tensors: keras.backend.concatenate((tensors[0], tensors[1])))

    Features_12 = concatenate([features_1, features_2])
    Features_13 = concatenate([features_1, features_3])
    Features_14 = concatenate([features_1, features_4])
    Features_23 = concatenate([features_2, features_3])
    Features_24 = concatenate([features_2, features_4])
    Features_34 = concatenate([features_3, features_4])

    dense1 = Dense(units=512, activation='relu', name='fc_1_OrderPrediction')

    RelationShip_12 = dense1(Features_12)
    RelationShip_13 = dense1(Features_13)
    RelationShip_14 = dense1(Features_14)
    RelationShip_23 = dense1(Features_23)
    RelationShip_24 = dense1(Features_24)
    RelationShip_34 = dense1(Features_34)

    #Concatenate_RelationShips = Lambda(lambda tensors: keras.backend.concatenate((tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5])))

    Features_Final = concatenate([RelationShip_12, RelationShip_13, RelationShip_14, RelationShip_23, RelationShip_24, RelationShip_34])

    prediction = Dense(units=12, activation='softmax', name='fc_final_OrderPrediction')(Features_Final)

    siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

    siamese_model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    siamese_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return siamese_model