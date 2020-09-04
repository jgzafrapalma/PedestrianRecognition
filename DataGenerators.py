import numpy as np
from tensorflow.keras.utils import Sequence
from FuncionesAuxiliares import ShuffleFrames
from tensorflow.keras.utils import to_categorical

import pickle

class DataGeneratorShuffle(Sequence):
    def __init__(self, list_IDs, path_instances, n_frames, batch_size=32, dim=(32, 32, 32), n_channels=1, n_clases=10, shuffle=True, normalized=True, step_swaps=5):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_clases = n_clases
        self.shuffle = shuffle
        self.n_frames = n_frames
        self.swaps = 0
        self.step_swaps = step_swaps
        self.normalized = normalized
        self.path_instances = path_instances
        self.n_epochs = 0 #Variable para llevar la cuenta de la epoca en la que se está
        self.on_epoch_end()

    def on_epoch_end(self):
        #Se crea un vector de indices(desde 0 hasta numero de identificadores menos 1)
        self.indexes = np.arange(len(self.list_IDs))  # ???? len(self.list_IDs)*2
        #Si shuffle se encuentra a True se cambia el orden de los elementos del vector de indices
        if self.shuffle:
            # Cambios el orden de entrada de los patrones en cada epoca
            np.random.shuffle(self.indexes)
        """Se incrementa el numero de swaps mientras que sea menor que el numero de frames. En el momento que el numero de
        swaps es igual al de frames, todos los frames serán intercambiados"""
        if self.swaps < self.n_frames:
            #El numero de swaps se incrementa cada step_swaps
            if self.n_epochs % self.step_swaps == 0:
                self.swaps += 1
        self.n_epochs += 1

    def __data_generation(self, list_IDs_temp):
        """Se reserva espacio para almacenar las instancias del batch actual y las etiquetas de esas instancias.
        En un mismo batch se almacenan los frames ordenados de una instancia y en la siguientes posición los mismos
        frames pero desordenados."""
        X = np.empty((self.batch_size, self.n_frames, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        i = 0 #Variable utilizada como contador del indice en el que se estan almacenando las intancias
        for ID_instance in list_IDs_temp:

            #print(ID_instance)

            #frames = extract_Frames_Matriz(self.path_instances, ID_instance, self.n_frames)

            with open(self.path_instances + '/' + ID_instance, 'rb') as input:
                instance = pickle.load(input)

            """Me quedo unicamente con el campo de frames de la instancia que almacena el vector de numpy con los fotograma
            de los vídeos"""
            frames = instance["frames"]

            #Normalización de los frames
            if self.normalized:
                frames = frames * 1 / 255

            #Se almacenan los frames ordenados y su etiqueta
            X[i, ] = frames
            y[i] = 0

            i += 1

            X[i, ] = ShuffleFrames(frames, self.swaps)
            y[i] = 1

            i += 1

        return X, to_categorical(y)

    def __getitem__(self, index):
        """Lista con los indices de las instancias que van a estar en el batch index. Por cada instancia, se va a tener
        la secuencia de frames ordenados y la secuencia de frames desordenados. Por lo tanto, el número de instancias
        seleccionadas será igual a la mitad del tamaño del batch (batch_size/2)"""
        indexes_batch = self.indexes[int(index * (self.batch_size / 2)):int((index + 1) * (self.batch_size / 2))]

        #Obtengo el identificador de las instancias que van a estar en el batch index
        list_IDs_temp = [self.list_IDs[k] for k in indexes_batch]

        #LLamo a la función para generar los datos con el identificador de las instancias que forman el batch
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __len__(self):
        return int(np.floor(len(self.list_IDs) * 2 / self.batch_size))
