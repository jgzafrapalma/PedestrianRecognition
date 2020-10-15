import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.append(os.path.join(rootdir, 'base_models'))

from base_models import CONV3D

from kerastuner import HyperModel

from tensorflow.keras.layers import Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



class HyperModel_Shuffle_CONV3D(HyperModel):
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

        x = basemodel(inputs, training=True)

        # Se definen las capas de clasificación del modelo
        x = Dropout(
                rate=hp.Float(
                    "dropout_rate_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05
                ),
                name='Dropout_1_Shuffle'
            )(x)

        features = Flatten(name='Flatten_Shuffle')(x)

        x = Dense(
                units=hp.Int(
                    "unit", min_value=32, max_value=512, step=32, default=64
                ),
                activation=hp.Choice(
                    "dense_activation", values=["relu", "tanh", "sigmoid"], default="relu"
                ),
                name='FC_1_Shuffle'
            )(features)

        x = Dropout(
                rate=hp.Float(
                    "dropout_rate_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05
                ),
                name='Dropout_2_Shuffle'
            )(x)

        outputs = Dense(self.num_classes, activation='softmax', name='FC_Final_Shuffle')(x)

        # Se define el modelo
        model = Model(inputs, outputs)

        model.summary()

        optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model