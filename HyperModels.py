from kerastuner import HyperModel
from tensorflow import keras

from tensorflow.keras.layers import Conv3D, Flatten, Dropout, Dense

class HyperModelShuffleConv3D(HyperModel):
    """Constructor de la clase, recibe las dimensiones de la entrada y el número de clases (salidas)"""
    def __init__(self, input_shape, num_classes):
        """Se inicializan las variables de la clase"""
        self.input_shape = input_shape
        self.num_classes = num_classes
    """Función en la que se define el modelo del que se quieren optimizar las hiperparámetros"""
    def build(self, hp):
        model = keras.Sequential()

        model.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                         activation='relu', input_shape=self.input_shape))

        model.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                         activation='relu'))

        model.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
        activation='relu'))

        model.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                         activation='relu'))

        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout_rate_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05
                )
            )
        )

        model.add(Flatten())

        model.add(
            Dense(
                units=hp.Int(
                    "unit", min_value=32, max_value=512, step=32, default=64
                ),
                activation=hp.Choice(
                    "dense_activation", values=["relu", "tanh", "sigmoid"], default="relu"
                )
            )
        )

        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout_rate_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05
                )
            )
        )

        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        optimizer = keras.optimizers.Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3
            )
        )

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model