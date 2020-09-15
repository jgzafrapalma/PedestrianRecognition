from kerastuner import HyperModel
from tensorflow import keras

from tensorflow.keras.layers import Conv3D, Flatten, Dropout, Dense

class HyperModel_Shuffle_CONV3D(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""
    def __init__(self, input_shape, num_classes):
        """Se inicializan las variables de la clase"""
        self.input_shape = input_shape
        self.num_classes = num_classes
    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""
    def build(self, hp):
        model = keras.Sequential()

        model.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                         activation='relu', input_shape=self.input_shape, name='conv3d_1_shuffle'))

        model.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                         activation='relu', name='conv3d_2_shuffle'))

        model.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
        activation='relu', name='conv3d_3_shuffle'))

        model.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                         activation='relu', name='conv3d_4_shuffle'))

        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout_rate_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05
                ),
                name='dropout_1_shuffle'
            )
        )

        model.add(Flatten(), name='flatten_shuffle')

        model.add(
            Dense(
                units=hp.Int(
                    "unit", min_value=32, max_value=512, step=32, default=64
                ),
                activation=hp.Choice(
                    "dense_activation", values=["relu", "tanh", "sigmoid"], default="relu"
                ),
                name='fc_1_shuffle'
            )
        )

        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout_rate_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05
                ),
                name='dropout_2_shuffle'
            )
        )

        model.add(Dense(self.num_classes, activation='softmax', name='fc_2_shuffle'))

        #model.summary()

        optimizer = keras.optimizers.Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model


#!!!!!!COMPROBAR SI ES NECESARIA PASAR EL PARAMETROS TRAINING A LAS CAPAS DE CONVOLUCIÓN!!!!!!!!!!!
"""Hipermodelo utilizado para la optimización de los hiperparámetros de la capa de clasificación durante la transferencia de información,
utilizando la tarea de pretexto Shuffle con un modelo de convolución 3D"""
class HyperModel_FINAL_Shuffle_CONV3D_Regression_CL(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""
    def __init__(self, input_shape, num_classes, path_weights):
        """Se inicializan las variables de la clase"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights
    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""
    def build(self, hp):
        model = keras.Sequential()

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

        model.add(conv3d_4_shuffle)

        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout_rate_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05
                ),
                name='dropout_1_final'
            )
        )

        model.add(Flatten(name='flatten_final'))

        model.add(
            Dense(
                units=hp.Int(
                    "unit", min_value=32, max_value=512, step=32, default=64
                ),
                activation=hp.Choice(
                    "dense_activation", values=["relu", "tanh", "sigmoid"], default="relu"
                ),
                name='fc_1_final'
            )
        )

        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout_rate_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05
                ),
                name='dropout_2_final'
            )
        )

        model.add(Dense(self.num_classes, activation='sigmoid', name='fc_2_final'))

        optimizer = keras.optimizers.Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.load_weights(self.path_weights, by_name=True)

        model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])

        return model

"""Hipermodelo utilizado para el ajuste del coeficiente de aprendizaje durante el ajuste fino."""
class HyperModel_FINAL_Shuffle_CONV3D_Regression_FT(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""
    def __init__(self, input_shape, num_classes, path_weights, hyperparameters):
        """Se inicializan las variables de la clase"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights
        self.hyperparameters = hyperparameters
    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""
    def build(self, hp):
        model = keras.Sequential()

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

        model.add(conv3d_4_shuffle)

        model.add(
            Dropout(
                rate=self.hyperparameters['dropout_rate_1'],
                name='dropout_1_final'
            )
        )

        model.add(Flatten(name='flatten_final'))

        model.add(
            Dense(
                units=self.hyperparameters['unit'],
                activation=self.hyperparameters['dense_activation'],
                name='fc_1_final'
            )
        )

        model.add(
            Dropout(
                rate=self.hyperparameters['dropout_rate_2'],
                name='dropout_2_final'
            )
        )

        model.add(Dense(self.num_classes, activation='sigmoid', name='fc_2_final'))

        optimizer = keras.optimizers.Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.load_weights(self.path_weights, by_name=True)

        model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])

        return model