import numpy as np
from tensorflow.keras.utils import Sequence
from FuncionesAuxiliares import extract_Frames_Matriz
from FuncionesAuxiliares import ShuffleFrames

from FuncionesAuxiliares import create_Train_Validation
from FuncionesAuxiliares import save_frames

from tensorflow.keras.utils import to_categorical

import os

class DataGenerator(Sequence):
    def __init__(self, list_IDs, path_instances, n_frames, batch_size=32, dim=(32, 32, 32), n_channels=1, n_clases=10, shuffle=True, normalized=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_clases = n_clases
        self.shuffle = shuffle
        self.n_frames = n_frames
        self.swaps = 0
        self.normalized=normalized
        self.path_instances = path_instances
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))  # ???? len(self.list_IDs)*2
        if self.shuffle:
            # Cambios el orden de entrada de los patrones en cada epoca
            np.random.shuffle(self.indexes)
        if self.swaps < self.n_frames:
            self.swaps += 1

    def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, self.n_frames, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        i = 0
        for ID_instance in list_IDs_temp:

            frames = extract_Frames_Matriz(self.path_instances, ID_instance, self.n_frames)

            if self.normalized:
                frames = frames * 1 / 255

            X[i, ] = frames
            y[i] = 0

            i += 1

            X[i, ] = ShuffleFrames(frames, self.swaps)
            y[i] = 1

            i += 1

        return X, to_categorical(y)

    def __getitem__(self, index):
        # El indice los videos que van a estar en el index-esimo lote
        # Como por cada video se tienen dos instancias, se van a seleccionar los indices de (batch_size/2) videos por
        # cada lote
        indexes_batch = self.indexes[int(index * (self.batch_size / 2)):int((index + 1) * (self.batch_size / 2))]

        #Obtengo el identificador de las instancias que forman el batch
        list_IDs_temp = [self.list_IDs[k] for k in indexes_batch]

        #LLamo a la función para generar los datos con el identificador de las instancias que forman el batch
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __len__(self):
        return int(np.floor(len(self.list_IDs) * 2 / self.batch_size))


if __name__ == '__main__':

    params = {'dim': (128, 128),
              'path_instances': '/media/jorge/DATOS/TFG/datasets/instances',
              'batch_size': 16,
              'n_clases': 2,
              'n_channels': 3,
              'n_frames': 8,
              'normalized': False,
              'shuffle': True}

    train_ids_instances, validation_id_instances = create_Train_Validation('/media/jorge/DATOS/TFG/datasets/instances', 0.3)

    generator = DataGenerator(train_ids_instances, **params)

    batch = generator.__getitem__(2)

    print(batch[0].shape)

    for i in range(len(batch[0])):
        save_frames('/media/jorge/DATOS/TFG/datasets/saves', batch[0][i], i)

