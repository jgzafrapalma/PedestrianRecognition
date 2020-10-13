from kerastuner import HyperModel
from tensorflow import keras

from models import CONV3D, CaffeNet

from tensorflow.keras.layers import Conv3D, Flatten, Dropout, Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



##########################################################################################################
###################################  PRETEXT TASK SHUFFLE  ###############################################
##########################################################################################################



class HyperModel_Shuffle_CONV3D(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""
    def __init__(self, input_shape, num_classes):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = input_shape
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



########################################################################################################################
##########################################  PRETEXT TASK ORDER PREDICTION  #############################################
########################################################################################################################



class HyperModel_OrderPrediction_SIAMESE(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""
    def __init__(self, the_input_shape, num_classes):
        """Se inicializan las variables de la clase"""
        self.the_input_shape = the_input_shape
        self.num_classes = num_classes
    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""
    def build(self, hp):

        # Se definen las 4 entradas del modelo
        input_1 = Input(shape=self.the_input_shape)
        input_2 = Input(shape=self.the_input_shape)
        input_3 = Input(shape=self.the_input_shape)
        input_4 = Input(shape=self.the_input_shape)

        base_model = CaffeNet(self.the_input_shape)

        # Las 4 entradas son pasadas a través del modelo base (calculo de las distintas convoluciones)
        output_1 = base_model(input_1)
        output_2 = base_model(input_2)
        output_3 = base_model(input_3)
        output_4 = base_model(input_4)

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

        siamese_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return siamese_model