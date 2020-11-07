from kerastuner import HyperModel

from tensorflow.keras.layers import Flatten, Dense, Input, concatenate, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, SGD

import tensorflow as tf

import numpy as np

"""Hipermodelo utilizado para la optimización de los hiperparámetros de la capa de clasificación durante la transferencia de información,
utilizando la tarea de pretexto OrderPrediction con un modelo siames"""
class HyperModel_OrderPrediction_SIAMESE_CrossingDetection_CL(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""

    def __init__(self, the_input_shape, num_classes, path_weights):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights

    """Función en la que se define el modelo del que se quieren optimizar los hiperparámetros"""

    def build(self, hp):

        # Se definen las 4 entradas del modelo
        input_1 = Input(shape=self.the_input_shape)
        input_2 = Input(shape=self.the_input_shape)
        input_3 = Input(shape=self.the_input_shape)
        input_4 = Input(shape=self.the_input_shape)

        #CaffeNet
        base_model = Sequential(name='CaffeNet')

        base_model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', data_format='channels_last',
                        activation='relu', input_shape=self.the_input_shape, name='Conv2D_1_CaffeNet'))
        base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_1_CaffeNet'))
        base_model.add(BatchNormalization())
        
        base_model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_2_CaffeNet'))
        base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_2_CaffeNet'))
        base_model.add(BatchNormalization())

        base_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_3_CaffeNet'))
        
        base_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_4_CaffeNet'))

        base_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_5_CaffeNet'))

        base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_3_CaffeNet'))

        #Se congelan las capas del modelo base
        base_model.trainable = False

        #Se cargan los pesos en las capas convoluciones para la transferencia de conocimiento
        with (self.path_weights).open('rb') as file_descriptor:
            conv_weights = np.load(file_descriptor, allow_pickle=True)

        base_model.set_weights(conv_weights)

        # Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
        output_1 = base_model(input_1, training=False)
        output_2 = base_model(input_2, training=False)
        output_3 = base_model(input_3, training=False)
        output_4 = base_model(input_4, training=False)

        flatten = Flatten(name='Flatten_1_FINAL')

        # Se obtienen los vectores de características de las 4 entradas
        features_1 = flatten(output_1)
        features_2 = flatten(output_2)
        features_3 = flatten(output_3)
        features_4 = flatten(output_4)

        # Capa densa utilizada para resumir las caracteristicas extraidas de las capas convolucionales para cada frame
        dense_1 = Dense(
            units=hp.Int(
                "units_dense_layers_1", min_value=512, max_value=4096, step=512, default=512
            ),
            activation='relu',
            name='FC_1_FINAL'
        )

        features_1 = dense_1(features_1)
        features_2 = dense_1(features_2)
        features_3 = dense_1(features_3)
        features_4 = dense_1(features_4)

        dropout_1 = Dropout(
            rate=hp.Float(
                "dropout_rate_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_1_FINAL'
        )

        features_1 = dropout_1(features_1)
        features_2 = dropout_1(features_2)
        features_3 = dropout_1(features_3)
        features_4 = dropout_1(features_4)

        Features_12 = concatenate([features_1, features_2])
        Features_13 = concatenate([features_1, features_3])
        Features_14 = concatenate([features_1, features_4])
        Features_23 = concatenate([features_2, features_3])
        Features_24 = concatenate([features_2, features_4])
        Features_34 = concatenate([features_3, features_4])

        # Capa densa que aprende la relación entre las características de los distintos fotogramas
        dense_2 = Dense(
            units=hp.Int(
                "units_dense_layers_2", min_value=512, max_value=4096, step=512, default=512
            ),
            activation='relu',
            name='FC_2_FINAL'
        )

        RelationShip_1_2 = dense_2(Features_12)
        RelationShip_1_3 = dense_2(Features_13)
        RelationShip_1_4 = dense_2(Features_14)
        RelationShip_2_3 = dense_2(Features_23)
        RelationShip_2_4 = dense_2(Features_24)
        RelationShip_3_4 = dense_2(Features_34)

        dropout_2 = Dropout(
            rate=hp.Float(
                "dropout_rate_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_2_FINAL'
        )

        RelationShip_1_2 = dropout_2(RelationShip_1_2)
        RelationShip_1_3 = dropout_2(RelationShip_1_3)
        RelationShip_1_4 = dropout_2(RelationShip_1_4)
        RelationShip_2_3 = dropout_2(RelationShip_2_3)
        RelationShip_2_4 = dropout_2(RelationShip_2_4)
        RelationShip_3_4 = dropout_2(RelationShip_3_4)


        # Concatenación de todas las relaciones
        Features_Final = concatenate(
            [RelationShip_1_2, RelationShip_1_3, RelationShip_1_4, RelationShip_2_3, RelationShip_2_4, RelationShip_3_4])

        prediction = Dense(units=self.num_classes, activation='softmax', name='FC_Final_FINAL')(Features_Final)

        siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

        siamese_model.summary()

        optimizer = SGD(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return siamese_model


"""Hipermodelo utilizado para el ajuste del coeficiente de aprendizaje durante el ajuste fino."""


class HyperModel_OrderPrediction_SIAMESE_CrossingDetection_FT(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""

    def __init__(self, the_input_shape, num_classes, path_weights, hyperparameters):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
        self.path_weights = path_weights
        self.hyperparameters = hyperparameters

    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""

    def build(self, hp):

        # Se definen las 4 entradas del modelo
        input_1 = Input(shape=self.the_input_shape)
        input_2 = Input(shape=self.the_input_shape)
        input_3 = Input(shape=self.the_input_shape)
        input_4 = Input(shape=self.the_input_shape)

        #CaffeNet
        base_model = Sequential()

        base_model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', data_format='channels_last',
                        activation='relu', input_shape=self.the_input_shape, name='Conv2D_1_CaffeNet'))
        base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_1_CaffeNet'))
        base_model.add(BatchNormalization())
        
        base_model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_2_CaffeNet'))
        base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_2_CaffeNet'))
        base_model.add(BatchNormalization())

        base_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_3_CaffeNet'))
        
        base_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_4_CaffeNet'))

        base_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_5_CaffeNet'))

        base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_3_CaffeNet'))

        # Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
        output_1 = base_model(input_1, training=False)
        output_2 = base_model(input_2, training=False)
        output_3 = base_model(input_3, training=False)
        output_4 = base_model(input_4, training=False)

        flatten_1 = Flatten(name='Flatten_1_FINAL')

        # Se obtienen los vectores de características de las 4 entradas
        features_1 = flatten_1(output_1)
        features_2 = flatten_1(output_2)
        features_3 = flatten_1(output_3)
        features_4 = flatten_1(output_4)

        # Capa densa utilizada para resumir las caracteristicas extraidas de las capas convolucionales para cada frame
        dense_1 = Dense(
            units=self.hyperparameters['units_dense_layers_1'],
            activation='relu',
            name='FC_1_FINAL'
        )

        features_1 = dense_1(features_1)
        features_2 = dense_1(features_2)
        features_3 = dense_1(features_3)
        features_4 = dense_1(features_4)

        dropout_1 = Dropout(
            rate=self.hyperparameters['dropout_rate_1'],
            name='Dropout_1_FINAL'
        )

        features_1 = dropout_1(features_1)
        features_2 = dropout_1(features_2)
        features_3 = dropout_1(features_3)
        features_4 = dropout_1(features_4)

        Features_12 = concatenate([features_1, features_2])
        Features_13 = concatenate([features_1, features_3])
        Features_14 = concatenate([features_1, features_4])
        Features_23 = concatenate([features_2, features_3])
        Features_24 = concatenate([features_2, features_4])
        Features_34 = concatenate([features_3, features_4])

        # Capa densa que aprende la relación entre las características de los distintos fotogramas
        dense_2 = Dense(
            units=self.hyperparameters['units_dense_layers_2'],
            activation='relu',
            name='FC_2_FINAL'
        )

        RelationShip_1_2 = dense_2(Features_12)
        RelationShip_1_3 = dense_2(Features_13)
        RelationShip_1_4 = dense_2(Features_14)
        RelationShip_2_3 = dense_2(Features_23)
        RelationShip_2_4 = dense_2(Features_24)
        RelationShip_3_4 = dense_2(Features_34)

        dropout_2 = Dropout(
            rate=self.hyperparameters['dropout_rate_2'],
            name='Dropout_2_FINAL'
        )

        RelationShip_1_2 = dropout_2(RelationShip_1_2)
        RelationShip_1_3 = dropout_2(RelationShip_1_3)
        RelationShip_1_4 = dropout_2(RelationShip_1_4)
        RelationShip_2_3 = dropout_2(RelationShip_2_3)
        RelationShip_2_4 = dropout_2(RelationShip_2_4)
        RelationShip_3_4 = dropout_2(RelationShip_3_4)

        # Concatenación de todas las relaciones
        Features_Final = concatenate(
            [RelationShip_1_2, RelationShip_1_3, RelationShip_1_4, RelationShip_2_3, RelationShip_2_4, RelationShip_3_4])

        prediction = Dense(units=self.num_classes, activation='softmax', name='FC_Final_FINAL')(Features_Final)

        siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

        siamese_model.summary()

        optimizer = SGD(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        with (self.path_weights).open('rb') as file_descriptor:
            siamese_model_weights = np.load(file_descriptor, allow_pickle=True)

        siamese_model.set_weights(siamese_model_weights)

        siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return siamese_model


class HyperModel_OrderPrediction_SIAMESE_CrossingDetection(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""

    def __init__(self, the_input_shape, num_classes):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes

    """Función en la que se define el modelo del que se quieren optimizar los hiperparámetros"""

    def build(self, hp):

        # Se definen las 4 entradas del modelo
        input_1 = Input(shape=self.the_input_shape)
        input_2 = Input(shape=self.the_input_shape)
        input_3 = Input(shape=self.the_input_shape)
        input_4 = Input(shape=self.the_input_shape)

        #CaffeNet
        base_model = Sequential(name='CaffeNet')

        base_model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', data_format='channels_last',
                        activation='relu', input_shape=self.the_input_shape, name='Conv2D_1_CaffeNet'))
        base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_1_CaffeNet'))
        base_model.add(BatchNormalization())
        
        base_model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_2_CaffeNet'))
        base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_2_CaffeNet'))
        base_model.add(BatchNormalization())

        base_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_3_CaffeNet'))
        
        base_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_4_CaffeNet'))

        base_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                        activation='relu', name='Conv2D_5_CaffeNet'))

        base_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_3_CaffeNet'))

        # Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
        output_1 = base_model(input_1)
        output_2 = base_model(input_2)
        output_3 = base_model(input_3)
        output_4 = base_model(input_4)

        flatten = Flatten(name='Flatten_1_FINAL')

        # Se obtienen los vectores de características de las 4 entradas
        features_1 = flatten(output_1)
        features_2 = flatten(output_2)
        features_3 = flatten(output_3)
        features_4 = flatten(output_4)

        # Capa densa utilizada para resumir las caracteristicas extraidas de las capas convolucionales para cada frame
        dense_1 = Dense(
            units=hp.Int(
                "units_dense_layers_1", min_value=512, max_value=4096, step=512, default=512
            ),
            activation='relu',
            name='FC_1_FINAL'
        )

        features_1 = dense_1(features_1)
        features_2 = dense_1(features_2)
        features_3 = dense_1(features_3)
        features_4 = dense_1(features_4)

        dropout_1 = Dropout(
            rate=hp.Float(
                "dropout_rate_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_1_FINAL'
        )

        features_1 = dropout_1(features_1)
        features_2 = dropout_1(features_2)
        features_3 = dropout_1(features_3)
        features_4 = dropout_1(features_4)

        Features_12 = concatenate([features_1, features_2])
        Features_13 = concatenate([features_1, features_3])
        Features_14 = concatenate([features_1, features_4])
        Features_23 = concatenate([features_2, features_3])
        Features_24 = concatenate([features_2, features_4])
        Features_34 = concatenate([features_3, features_4])

        # Capa densa que aprende la relación entre las características de los distintos fotogramas
        dense_2 = Dense(
            units=hp.Int(
                "units_dense_layers_2", min_value=512, max_value=4096, step=512, default=512
            ),
            activation='relu',
            name='FC_2_FINAL'
        )

        RelationShip_1_2 = dense_2(Features_12)
        RelationShip_1_3 = dense_2(Features_13)
        RelationShip_1_4 = dense_2(Features_14)
        RelationShip_2_3 = dense_2(Features_23)
        RelationShip_2_4 = dense_2(Features_24)
        RelationShip_3_4 = dense_2(Features_34)

        dropout_2 = Dropout(
            rate=hp.Float(
                "dropout_rate_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05
            ),
            name='Dropout_2_FINAL'
        )

        RelationShip_1_2 = dropout_2(RelationShip_1_2)
        RelationShip_1_3 = dropout_2(RelationShip_1_3)
        RelationShip_1_4 = dropout_2(RelationShip_1_4)
        RelationShip_2_3 = dropout_2(RelationShip_2_3)
        RelationShip_2_4 = dropout_2(RelationShip_2_4)
        RelationShip_3_4 = dropout_2(RelationShip_3_4)


        # Concatenación de todas las relaciones
        Features_Final = concatenate(
            [RelationShip_1_2, RelationShip_1_3, RelationShip_1_4, RelationShip_2_3, RelationShip_2_4, RelationShip_3_4])

        prediction = Dense(units=self.num_classes, activation='softmax', name='FC_Final_FINAL')(Features_Final)

        siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

        siamese_model.summary()

        optimizer = SGD(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            ),
            momentum=hp.Choice("momentum", [0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
        )

        siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return siamese_model