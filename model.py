from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def create_model(the_input_shape, dropout_rate, learning_rate):

    model = Sequential()

    model.add(Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                     activation='relu', input_shape=the_input_shape))

    model.add(Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu'))

    model.add(Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu'))

    model.add(Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last',
              activation='relu'))

    model.add(Dropout(dropout_rate))

    model.add(Flatten())

    model.add(Dense(64))

    model.add(Dropout(dropout_rate))

    model.add(Dense(2, activation='softmax'))

    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
