from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, Flatten, Dropout, Dense, Conv2D, MaxPooling2D, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras






##########################################################################################################
############################################  BASE MODELS  ###############################################
##########################################################################################################



class CaffeNet(Model):

    def __init__(self, input_shape):
        super(CaffeNet, self).__init__()

        #1th Convolutional Layers
        self.Conv2D_1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', data_format='channels_last',
                     activation='relu', input_shape=input_shape, name='Conv2D_1_CaffeNet')
        self.MaxPooling2D_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_1_CaffeNet')
        self.BatchNormalization_1 = BatchNormalization(name='BatchNormalization_1_CaffeNet')

        #2th Convolutional Layers
        self.Conv2D_2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last',
                     activation='relu', name='Conv2D_2_CaffeNet')
        self.MaxPooling2D_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_2_CaffeNet')
        self.BatchNormalization_2 = BatchNormalization(name='BatchNormalization_2_CaffeNet')

        #3th Convolutional Layers
        self.Conv2D_3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_3_CaffeNet')

        #4th Convolutional Layers
        self.Conv2D_4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_4_CaffeNet')

        #5th Convolutional Layers
        self.Conv2D_5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_5_CaffeNet')
        self.MaxPooling2D_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_3_CaffeNet')

    def call(self, inputs, training=True):
        #1th Convolutional Layers
        x = self.Conv2D_1(inputs)
        x = self.MaxPooling2D_1(x)
        x = self.BatchNormalization(x, training)

        #2th Convolutional Layers
        x = self.Conv2D_2(x)
        x = self.MaxPooling2D_2(x)
        x = self.BatchNormalization_2(x, training)

        #3th Convolutional Layers
        x = self.Conv2D_3(x)

        #4th Convolutional Layers
        x = self.Conv2D_4(x)

        #5th Convolutional Layers
        x = self.Conv2D_5(x)
        outputs = self.MaxPooling2D_3(x)

        return outputs



class CONV3D(Model):

    def __init__(self, input_shape):
        super(CONV3D, self).__init__()

        #1th Convolutional Layer
        self.Conv3D_1 = Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                    activation='relu', input_shape=input_shape, name='Conv3D_1_CONV3D')

        #2th Convolutional Layer
        self.Conv3D_2 = Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                    activation='relu', name='Conv3D_2_CONV3D')



"""def basemodel_CaffeNet(the_input_shape):

    basemodel = Sequential()

    #1th Convolutional Layer
    basemodel.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', data_format='channels_last',
                     activation='relu', input_shape=the_input_shape, name='conv2d_1_CaffeNet'))
    basemodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))
    basemodel.add(BatchNormalization())

    #2th Convolutional Layer
    basemodel.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last',
                     activation='relu', name='conv2d_2_CaffeNet'))
    basemodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))
    basemodel.add(BatchNormalization())

    #3th Convolutional Layer
    basemodel.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                     activation='relu', name='conv2d_3_CaffeNet'))

    #4th Convolutional Layer
    basemodel.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                     activation='relu', name='conv2d_4_CaffeNet'))

    #5th Convolutional Layer
    basemodel.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                     activation='relu', name='conv2d_5_CaffeNet'))
    basemodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))

    return basemodel"""



def basemodel_CONV3D(the_input_shape):

    basemodel = Sequential()

    basemodel.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu', input_shape=the_input_shape, name='conv3d_1_Conv3D'))

    basemodel.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu', name='conv3d_2_Conv3D'))

    basemodel.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu', name='conv3d_3_Conv3D'))

    basemodel.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu', name='conv3d_1_Conv3D'))

    return basemodel



##########################################################################################################
###################################  PRETEXT TASK SHUFFLE  ###############################################
##########################################################################################################


#Modelo para la tarea de pretexto de reconocer frames desordenados
def model_Shuffle_CONV3D(the_input_shape, dropout_rate_1, dropout_rate_2, dense_activation, units_dense_layer, learning_rate):


    #Se define la entrada del modelo
    inputs = Input(the_input_shape)

    #Se declara el modelo base que se va a emplear (capas convoluciones del modelo)
    x = basemodel_CONV3D(inputs, training=True)

    #Se definen las capas de clasificación del modelo
    x = Dropout(dropout_rate_1, name='dropout_1_shuffle')(x)

    features = Flatten(name='flatten_shuffle')(x)

    x = Dense(units=units_dense_layer, activation=dense_activation, name='fc_1_shuffle')(features)

    x = Dropout(dropout_rate_2, name='dropout_2_shuffle')(x)

    outputs = Dense(2, activation='softmax', name='fc_2_shuffle')(x)

    #Se define el modelo
    model = Model(inputs, outputs)

    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

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




##########################################################################################################
##################################  PRETEXT TASK ORDER PREDICTION  #######################################
##########################################################################################################



def model_OrderPrediction_SIAMESE(the_input_shape, learning_rate):

    
    #Se definen las 4 entradas del modelo
    input_1 = Input(shape=the_input_shape)
    input_2 = Input(shape=the_input_shape)
    input_3 = Input(shape=the_input_shape)
    input_4 = Input(shape=the_input_shape)

    """
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
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))"""

    base_model = CaffeNet(the_input_shape)

    #Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
    output_1 = base_model(input_1)
    output_2 = base_model(input_2)
    output_3 = base_model(input_3)
    output_4 = base_model(input_4)

    flatten_1 = Flatten(name='Flatten_1_OrderPrediction')

    #Se obtienen los vectores de características de las 4 entradas
    features_1 = flatten_1(output_1)
    features_2 = flatten_1(output_2)
    features_3 = flatten_1(output_3)
    features_4 = flatten_1(output_4)

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