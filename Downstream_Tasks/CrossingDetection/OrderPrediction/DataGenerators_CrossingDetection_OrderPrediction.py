import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import pickle
import math


class DataGeneratorCrossingDetectionOrderPrediction(Sequence):
    def __init__(self, list_IDs, path_instances, batch_size=32, dim=(128, 128, 32), n_channels=2, n_clases=1, shuffle=True, normalized=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_clases
        self.shuffle = shuffle
        self.normalized = normalized
        self.path_instances = path_instances
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        """Se reserva espacio para almacenar las instancias del batch actual y las etiquetas de esas instancias."""
        X1 = np.empty((len(list_IDs_temp), *self.dim, self.n_channels))
        X2 = np.empty((len(list_IDs_temp), *self.dim, self.n_channels))
        X3 = np.empty((len(list_IDs_temp), *self.dim, self.n_channels))
        X4 = np.empty((len(list_IDs_temp), *self.dim, self.n_channels))
        y = np.empty(len(list_IDs_temp), dtype=int)

        for i, ID_instance in enumerate(list_IDs_temp):

            with (self.path_instances / ID_instance).open('rb') as input:
                instance = pickle.load(input)

            frames = instance['frames']

            label = instance['crossing']

            #Normalización de los frames
            if self.normalized:
                frames = frames * 1 / 255

            X1[i, ] = frames[0]
            X2[i, ] = frames[1]
            X3[i, ] = frames[2]
            X4[i, ] = frames[3]
            y[i] = label

        return [X1, X2, X3, X4], to_categorical(y, num_classes=self.n_classes)

    def __getitem__(self, index):

        indexes_batch = self.indexes[int(index * self.batch_size):int((index + 1) * self.batch_size)]

        #Obtengo el identificador de las instancias que van a estar en el batch index
        list_IDs_temp = [self.list_IDs[k] for k in indexes_batch]

        #LLamo a la función para generar los datos con el identificador de las instancias que forman el batch
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def get_ID_instances_and_labels(self):

        #Obtengo el identificador de las instancias en el orden en el que son generadas
        ID_instances = [self.list_IDs[k] for k in self.indexes]

        real_labels = []
        for ID_instance in ID_instances:

            with (self.path_instances / ID_instance).open('rb') as input:
                instance = pickle.load(input)

            real_labels.append(instance['crossing'])

        return ID_instances, to_categorical(real_labels, num_classes=self.n_classes)

    def __len__(self):
        return int(math.ceil(len(self.list_IDs) / self.batch_size))