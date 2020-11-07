from kerastuner import HyperModel

from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import Sequential

import tensorflow as tf

import numpy as np


"""Hipermodelo utilizado para la optimización de los hiperparámetros de la capa de clasificación durante la transferencia de información,
utilizando la tarea de pretexto Shuffle con un modelo de convolución 3D"""
class HyperModel_Shuffle_CONV3D_CrossingDetection_CL(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""

    def __init__(self, the_input_shape, num_classes, path_weights):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights

    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""

    def build(self, hp):

        # Se define la entrada del modelo
        inputs = Input(self.the_input_shape)

        base_model = Sequential(name='CONV3D')

        base_model.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', input_shape=self.the_input_shape, name='Conv3D_1_CONV3D'))

        base_model.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_2_CONV3D'))

        base_model.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_3_CONV3D'))

        base_model.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_4_CONV3D'))

        #Se congelan las capas del modelo base
        base_model.trainable = False

        #Se cargan los pesos en las capas convoluciones para la transferencia de conocimiento
        with (self.path_weights).open('rb') as file_descriptor:
            conv_weights = np.load(file_descriptor, allow_pickle=True)

        base_model.set_weights(conv_weights)

        output_1 = base_model(inputs)

        x = Dropout(
            rate=hp.Float(
                "dropout_rate_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_1_FINAL'
        )(output_1)

        x = Flatten(name='Flatten_FINAL')(x)

        x = Dense(
            units=hp.Int(
                "unit", min_value=32, max_value=512, step=32, default=64
            ),
            activation='relu',
            name='FC_1_FINAL'
        )(x)

        x = Dropout(
            rate=hp.Float(
                "dropout_rate_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_2_FINAL'
        )(x)

        outputs = Dense(self.num_classes, activation='softmax', name='FC_2_FINAL')(x)

        model = Model(inputs, outputs)

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.load_weights(str(self.path_weights), by_name=True)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model


"""Hipermodelo utilizado para el ajuste del coeficiente de aprendizaje durante el ajuste fino."""


class HyperModel_Shuffle_CONV3D_CrossingDetection_FT(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""

    def __init__(self, the_input_shape, num_classes, path_weights, hyperparameters):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights
        self.hyperparameters = hyperparameters

    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""

    def build(self, hp):

        # Se define la entrada del modelo
        inputs = Input(self.the_input_shape)

        base_model = Sequential(name='CONV3D')

        base_model.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', input_shape=self.the_input_shape, name='Conv3D_1_CONV3D'))

        base_model.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_2_CONV3D'))

        base_model.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_3_CONV3D'))

        base_model.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_4_CONV3D'))

        # El modelo base se pone en modo inferencia
        output_1 = base_model(inputs, training=False)

        x = Dropout(
            rate=self.hyperparameters['dropout_rate_1'],
            name='Dropout_1_FINAL'
        )(output_1)

        x = Flatten(name='Flatten_FINAL')(x)

        x = Dense(
            units=self.hyperparameters['unit'],
            activation='relu',
            name='FC_1_FINAL'
        )(x)

        x = Dropout(
            rate=self.hyperparameters['dropout_rate_2'],
            name='Dropout_2_FINAL'
        )(x)

        outputs = Dense(self.num_classes, activation='softmax', name='FC_2_FINAL')(x)

        model = Model(inputs, outputs)

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        with (self.path_weights).open('rb') as file_descriptor:
            model_weights = np.load(file_descriptor, allow_pickle=True)

        model.set_weights(model_weights)

        model.load_weights(str(self.path_weights), by_name=True)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model



class HyperModel_Shuffle_CONV3D_CrossingDetection(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""

    def __init__(self, the_input_shape, num_classes):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""

    def build(self, hp):

        # Se define la entrada del modelo
        inputs = Input(self.the_input_shape)

        base_model = Sequential(name='CONV3D')

        base_model.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', input_shape=self.the_input_shape, name='Conv3D_1_CONV3D'))

        base_model.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_2_CONV3D'))

        base_model.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_3_CONV3D'))

        base_model.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name='Conv3D_4_CONV3D'))

        # El modelo base se pone en modo inferencia
        output_1 = base_model(inputs)

        x = Dropout(
            rate=hp.Float(
                "dropout_rate_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_1_FINAL'
        )(output_1)

        x = Flatten(name='Flatten_FINAL')(x)

        x = Dense(
            units=hp.Int(
                "unit", min_value=32, max_value=512, step=32, default=64
            ),
            activation='relu',
            name='FC_1_FINAL'
        )(x)

        x = Dropout(
            rate=hp.Float(
                "dropout_rate_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_2_FINAL'
        )(x)

        outputs = Dense(self.num_classes, activation='softmax', name='FC_2_FINAL')(x)

        model = Model(inputs, outputs)

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model




"""Hipermodelo utilizado para la optimización de los hiperparámetros de la capa de clasificación durante la transferencia de información,
utilizando la tarea de pretexto Shuffle con un modelo de convolución 3D"""
class HyperModel_Shuffle_C3D_CrossingDetection_CL(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""

    def __init__(self, the_input_shape, num_classes, path_weights):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights

    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""

    def build(self, hp):

        # Se define la entrada del modelo
        inputs = Input(self.the_input_shape)

        # Se declara el modelo base que se va a emplear (capas convoluciones del modelo)
        basemodel = C3D(self.the_input_shape)

        # Se congela el modelo base para que sus pesos no sean entrenables
        basemodel.trainable = False

        # El modelo base se pone en modo inferencia
        x = basemodel(inputs, training=False)

        features = Flatten(name='Flatten_FINAL')(x)

        x = Dense(
            units=hp.Int(
                "units_dense_layers_1", min_value=512, max_value=4096, step=512, default=512
            ),
            activation='relu',
            name='FC_1_FINAL'
        )(features)        

        x = Dropout(
            rate=hp.Float(
                "dropout_rate_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_1_FINAL'
        )(x)

        x = Dense(
            units=hp.Int(
                "units_dense_layers_2", min_value=512, max_value=4096, step=512, default=512
            ),
            activation='relu',
            name='FC_2_FINAL'
        )(x)

        x = Dropout(
            rate=hp.Float(
                "dropout_rate_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_2_FINAL'
        )(x)

        outputs = Dense(self.num_classes, activation='softmax', name='FC_3_FINAL')(x)

        model = Model(inputs, outputs)

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.load_weights(str(self.path_weights), by_name=True)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model


"""Hipermodelo utilizado para el ajuste del coeficiente de aprendizaje durante el ajuste fino."""


class HyperModel_Shuffle_C3D_CrossingDetection_FT(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""

    def __init__(self, the_input_shape, num_classes, path_weights, hyperparameters):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights
        self.hyperparameters = hyperparameters

    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""

    def build(self, hp):

        # Se define la entrada del modelo
        inputs = Input(self.the_input_shape)

        # Se declara el modelo base que se va a emplear (capas convoluciones del modelo)
        basemodel = C3D(self.the_input_shape)

        # Se descongela el modelo base
        basemodel.trainable = True

        # El modelo base se pone en modo inferencia
        x = basemodel(inputs, training=False)

        features = Flatten(name='Flatten_FINAL')(x)

        x = Dense(
            units=self.hyperparameters['units_dense_layers_1'],
            activation='relu',
            name='FC_1_FINAL'
        )(features)        

        x = Dropout(
            rate=self.hyperparameters['dropout_rate_1'],
            name='Dropout_1_FINAL'
        )(x)

        x = Dense(
            units=self.hyperparameters['units_dense_layers_2'],
            activation='relu',
            name='FC_2_FINAL'
        )(x)

        x = Dropout(
            rate=self.hyperparameters['dropout_rate_2'],
            name='Dropout_2_FINAL'
        )(x)

        outputs = Dense(self.num_classes, activation='softmax', name='FC_3_FINAL')(x)

        model = Model(inputs, outputs)

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.load_weights(str(self.path_weights), by_name=True)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model