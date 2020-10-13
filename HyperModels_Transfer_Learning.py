from kerastuner import HyperModel
from models import CONV3D, CaffeNet

from tensorflow.keras.layers import Flatten, Dropout, Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



##########################################################################################################
###################################  PRETEXT TASK SHUFFLE  ###############################################
##########################################################################################################



"""Hipermodelo utilizado para la optimización de los hiperparámetros de la capa de clasificación durante la transferencia de información,
utilizando la tarea de pretexto Shuffle con un modelo de convolución 3D"""
class HyperModel_FINAL_Shuffle_CONV3D_CrossingDetection_CL(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""

    def __init__(self, the_input_shape, num_classes, path_weights):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights

    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""

    def build(self, hp):
        """model = keras.Sequential()

        conv3d_1_shuffle = Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                  activation='relu', input_shape=self.input_shape, name='conv3d_1_shuffle')

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

        model.add(conv3d_4_shuffle)"""

        # Se define la entrada del modelo
        inputs = Input(self.the_input_shape)

        # Se declara el modelo base que se va a emplear (capas convoluciones del modelo)
        basemodel = CONV3D(self.the_input_shape)

        # Se congela el modelo base para que sus pesos no sean entrenables
        basemodel.trainable = False

        # El modelo base se pone en modo inferencia
        x = basemodel(inputs, training=False)

        x = Dropout(
            rate=hp.Float(
                "dropout_rate_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_1_FINAL'
        )(x)

        x = Flatten(name='Flatten_FINAL')(x)

        x = Dense(
            units=hp.Int(
                "unit", min_value=32, max_value=512, step=32, default=64
            ),
            activation=hp.Choice(
                "dense_activation", values=["relu", "tanh", "sigmoid"], default="relu"
            ),
            name='FC_1_FINAL'
        )(x)

        x = Dropout(
            rate=hp.Float(
                "dropout_rate_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_2_FINAL'
        )(x)

        outputs = Dense(self.num_classes, activation='softmax', name='FC_2_Final')(x)

        model = Model(inputs, outputs)

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.load_weights(self.path_weights, by_name=True)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model


"""Hipermodelo utilizado para el ajuste del coeficiente de aprendizaje durante el ajuste fino."""


class HyperModel_FINAL_Shuffle_CONV3D_CrossingDetection_FT(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""

    def __init__(self, the_input_shape, num_classes, path_weights, hyperparameters):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights
        self.hyperparameters = hyperparameters

    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""

    def build(self, hp):
        """model = keras.Sequential()

        conv3d_1_shuffle = Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                  activation='relu', input_shape=self.input_shape, name='conv3d_1_shuffle')

        model.add(conv3d_1_shuffle)

        conv3d_2_shuffle = Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                  activation='relu', name='conv3d_2_shuffle')

        model.add(conv3d_2_shuffle)

        conv3d_3_shuffle = Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                  activation='relu', name='conv3d_3_shuffle')

        model.add(conv3d_3_shuffle)

        conv3d_4_shuffle = Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                  activation='relu', name='conv3d_4_shuffle')

        model.add(conv3d_4_shuffle)"""

        # Se define la entrada del modelo
        inputs = Input(self.the_input_shape)

        # Se declara el modelo base que se va a emplear (capas convoluciones del modelo)
        basemodel = CONV3D(self.the_input_shape)

        # Se congela el modelo base para que sus pesos no sean entrenables
        basemodel.trainable = True

        # El modelo base se pone en modo inferencia
        x = basemodel(inputs, training=False)

        x = Dropout(
            rate=self.hyperparameters['dropout_rate_1'],
            name='Dropout_1_FINAL'
        )(x)

        x = Flatten(name='Flatten_FINAL')(x)

        x = Dense(
            units=self.hyperparameters['unit'],
            activation=self.hyperparameters['dense_activation'],
            name='FC_1_FINAL'
        )(x)

        x = Dropout(
            rate=self.hyperparameters['dropout_rate_2'],
            name='Dropout_2_FINAL'
        )(x)

        outputs = Dense(self.num_classes, activation='sigmoid', name='FC_2_Final')(x)

        model = Model(inputs, outputs)

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.load_weights(self.path_weights, by_name=True)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model



########################################################################################################################
##########################################  PRETEXT TASK ORDER PREDICTION  #############################################
########################################################################################################################



class HyperModel_FINAL_OrderPrediction_SIAMESE_CrossingDetection_CL(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""
    def __init__(self, the_input_shape, num_classes, path_weights):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights
    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""
    def build(self, hp):

        # Se definen las 4 entradas del modelo
        input_1 = Input(shape=self.the_input_shape)
        input_2 = Input(shape=self.the_input_shape)
        input_3 = Input(shape=self.the_input_shape)
        input_4 = Input(shape=self.the_input_shape)

        base_model = CaffeNet(self.the_input_shape)

        base_model.trainable = False

        # Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
        output_1 = base_model(input_1, training=False)
        output_2 = base_model(input_2, training=False)
        output_3 = base_model(input_3, training=False)
        output_4 = base_model(input_4, training=False)

        flatten_1 = Flatten(name='Flatten_1_OrderPrediction')

        # Se obtienen los vectores de características de las 4 entradas
        features_1 = flatten_1(output_1)
        features_2 = flatten_1(output_2)
        features_3 = flatten_1(output_3)
        features_4 = flatten_1(output_4)

        Features_12 = concatenate([features_1, features_2])
        Features_13 = concatenate([features_1, features_3])
        Features_14 = concatenate([features_1, features_4])
        Features_23 = concatenate([features_2, features_3])
        Features_24 = concatenate([features_2, features_4])
        Features_34 = concatenate([features_3, features_4])

        dense1 = Dense(
            units=hp.Int(
                "unit", min_value=32, max_value=512, step=32, default=64
            ),
            activation=hp.Choice(
                "dense_activation", values=["relu", "tanh", "sigmoid"], default="relu"
            ),
            name='FC_1_OrderPrediction'
        )

        RelationShip_12 = dense1(Features_12)
        RelationShip_13 = dense1(Features_13)
        RelationShip_14 = dense1(Features_14)
        RelationShip_23 = dense1(Features_23)
        RelationShip_24 = dense1(Features_24)
        RelationShip_34 = dense1(Features_34)

        Features_Final = concatenate([RelationShip_12, RelationShip_13, RelationShip_14, RelationShip_23, RelationShip_24, RelationShip_34])

        prediction = Dense(units=self.num_classes, activation='softmax', name='FC_Final_OrderPrediction')(Features_Final)

        siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

        siamese_model.summary()

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        siamese_model.load_weights(self.path_weights, by_name=True)

        siamese_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return siamese_model