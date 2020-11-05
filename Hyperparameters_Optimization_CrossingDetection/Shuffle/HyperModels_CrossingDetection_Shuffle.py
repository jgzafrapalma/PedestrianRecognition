import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.append(os.path.join(rootdir, 'base_models'))

from base_models import CONV3D, C3D

from kerastuner import HyperModel

from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf



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

        # Se declara el modelo base que se va a emplear (capas convoluciones del modelo)
        basemodel = CONV3D(self.the_input_shape)

        # El modelo base se pone en modo inferencia
        x = basemodel(inputs)

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