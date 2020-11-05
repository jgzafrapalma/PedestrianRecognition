import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
rootdir = os.path.dirname(parentparentdir)
sys.path.append(os.path.join(rootdir, 'base_models'))

from base_models import CaffeNet

from kerastuner import HyperModel

from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



"""Hipermodelo utilizado para la optimización de los hiperparámetros de la capa de clasificación durante la transferencia de información,
utilizando la tarea de pretexto OrderPrediction con un modelo siames"""
class HyperModel_OrderPrediction_SIAMESE_CrossingDetection_CL(HyperModel):
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

        basemodel = CaffeNet(self.the_input_shape)

        basemodel.trainable = False

        # Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
        output_1 = basemodel(input_1, training=False)
        output_2 = basemodel(input_2, training=False)
        output_3 = basemodel(input_3, training=False)
        output_4 = basemodel(input_4, training=False)

        flatten_1 = Flatten(name='Flatten_1_FINAL')

        # Se obtienen los vectores de características de las 4 entradas
        features_1 = flatten_1(output_1)
        features_2 = flatten_1(output_2)
        features_3 = flatten_1(output_3)
        features_4 = flatten_1(output_4)

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

        # Concatenación de todas las relaciones
        Features_Final = concatenate(
            [RelationShip_1_2, RelationShip_1_3, RelationShip_1_4, RelationShip_2_3, RelationShip_2_4, RelationShip_3_4])

        prediction = Dense(units=self.num_classes, activation='softmax', name='FC_Final_FINAL')(Features_Final)

        siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

        siamese_model.summary()

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        siamese_model.load_weights(str(self.path_weights), by_name=True)

        siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

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

        basemodel = CaffeNet(self.the_input_shape)

        basemodel.trainable = True

        # Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
        output_1 = basemodel(input_1, training=False)
        output_2 = basemodel(input_2, training=False)
        output_3 = basemodel(input_3, training=False)
        output_4 = basemodel(input_4, training=False)

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

        # Concatenación de todas las relaciones
        Features_Final = concatenate(
            [RelationShip_1_2, RelationShip_1_3, RelationShip_1_4, RelationShip_2_3, RelationShip_2_4, RelationShip_3_4])

        prediction = Dense(units=self.num_classes, activation='softmax', name='FC_Final_FINAL')(Features_Final)

        siamese_model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=prediction)

        siamese_model.summary()

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        siamese_model.load_weights(str(self.path_weights), by_name=True)

        siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return siamese_model
