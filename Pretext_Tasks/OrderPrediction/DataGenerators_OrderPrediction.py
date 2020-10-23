from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
import pickle

from pathlib import Path
from os.path import join, splitext
from random import shuffle as sf

class DataGeneratorOrderPrediction(Sequence):
    def __init__(self, list_IDs, path_instances, batch_size=32, dim=(32, 32), n_channels=1, n_clases=10, shuffle=True, normalized=True, n_epochs=100):
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_clases
        self.normalized = normalized
        self.shuffle = shuffle
        #Ruta donde se encuentran las instancias
        self.path_instances = path_instances

        #Lista con el nombre de los peatones que forman parte del conjunto de entrenamiento
        self.list_IDs = list_IDs

        self.list_IDs_epoch = list()

        for _ in range(6):

            aux = self.list_IDs.copy()

            self.list_IDs_epoch.extend(aux)


        #Lista con los peatones disponibles a la hora de ir rellenando los batches
        #self.restant_pedestrians = self.list_IDs.copy()

        #Se crea un diccionario utilizado para ver las clases que quedan por ser seleccionadas de cada peatón
        self.restant_classes_pedestrians = dict([(ped, [0, 1, 2, 3, 4, 5]) for ped in self.list_IDs])

        self.curriculum_learning_level = 1

        #Número de epocas necesarías para pasar el nivel 1 de dificultad de puzzle al nivel 2
        self.change_level1_to_level2 = n_epochs * 25
        # Número de epocas necesarías para pasar el nivel 2 de dificultad de puzzle al nivel 3
        self.change_level2_to_level3 = n_epochs * 35

        self.n_epochs = -1 #Llevar la cuenta del número de epocas que se han ejecutado

        self.on_epoch_end()

    def on_epoch_end(self):

        self.n_epochs += 1 # Número de epocas que se llevan ejecutadas

        #Se comprueba si han pasado el número de epocas necesarias para pasar del nivel 1 al nivel 2, para la siguiente época
        if self.n_epochs == self.change_level1_to_level2:
            self.restant_classes_pedestrians = dict([(ped, [6, 7, 8, 9]) for ped in self.list_IDs])
            self.curriculum_learning_level = 2
        elif self.n_epochs == self.change_level2_to_level3:
            self.restant_classes_pedestrians = dict([(ped, [10, 11]) for ped in self.list_IDs])
            self.curriculum_learning_level = 3

        #Al terminar una época la lista de restances debería de volver a ser reasignada con todos los peatones de entrenamiento

        #self.restant_pedestrians = self.list_IDs.copy()

        self.list_IDs_epoch = list()

        for _ in range(6):

            aux = self.list_IDs.copy()

            self.list_IDs_epoch.extend(aux)


    def __data_generation(self, list_IDs_temp):

        """Se reserva espacio para almacenar las instancias del batch actual y las etiquetas de esas instancias."""
        X1 = np.empty((len(list_IDs_temp)*2, *self.dim, self.n_channels))
        X2 = np.empty((len(list_IDs_temp)*2, *self.dim, self.n_channels))
        X3 = np.empty((len(list_IDs_temp)*2, *self.dim, self.n_channels))
        X4 = np.empty((len(list_IDs_temp)*2, *self.dim, self.n_channels))
        y = np.empty(len(list_IDs_temp)*2, dtype=int)


        #Indice en del batch en el que se van a escribir las instancias
        index_batch = 0

        for ped in list_IDs_temp:

            ped_noext = splitext(ped)[0]

            print(self.restant_classes_pedestrians[ped])

            #Del peatón seleccionado se selecciona de manera aleatoria una de sus clases
            pos_class = np.random.randint(len(self.restant_classes_pedestrians[ped]))

            clas = self.restant_classes_pedestrians[ped][pos_class]

            self.restant_classes_pedestrians[ped].remove(clas)

            PathInstance = Path(join(self.path_instances, ped_noext))

            #Se carga del disco duro la instancia correspondiente al peatón 'ped' y la clase 'clas' y se almacena en el batch

            with (PathInstance / (ped_noext + '_' + str(clas) + 'p.pkl')).open('rb') as input:
                instance1 = pickle.load(input)

            frames = instance1['frames']

            #Normalización de los frames
            if self.normalized:
                frames = frames * 1 / 255

            X1[index_batch, ] = frames[0]
            X2[index_batch, ] = frames[1]
            X3[index_batch, ] = frames[2]
            X4[index_batch, ] = frames[3]

            y[index_batch, ] = instance1['class']

            index_batch += 1

            #Del peatón seleccionado se selecciona de manera aleatoria una de sus clases
            pos_class = np.random.randint(len(self.restant_classes_pedestrians[ped]))

            clas = self.restant_classes_pedestrians[ped][pos_class]

            self.restant_classes_pedestrians[ped].remove(clas)

            with (PathInstance / (ped_noext + '_' + str(clas) + 'p.pkl')).open('rb') as input:
                instance2 = pickle.load(input)

            frames = instance2['frames']

            #Normalización de los frames
            if self.normalized:
                frames = frames * 1 / 255

            X1[index_batch, ] = frames[0]
            X2[index_batch, ] = frames[1]
            X3[index_batch, ] = frames[2]
            X4[index_batch, ] = frames[3]

            y[index_batch, ] = instance2['class']

            index_batch += 1

        #Para que los elementos del batch sean introducidos a la red de forma desordenada (que no esten correlativos los
        #elementos pertenecientes a la misma instancia)
        if self.shuffle:
            shuffler = np.random.permutation(len(X1))
            X1 = X1[shuffler]
            X2 = X2[shuffler]
            X3 = X3[shuffler]
            X4 = X4[shuffler]
            y = y[shuffler]

        return [X1, X2, X3, X4], to_categorical(y, self.n_classes)

    def __getitem__(self, index):
        """Lista con los indices de las instancias que van a estar en el batch index. Por cada instancia, se va a tener
        la secuencia de frames ordenados y la secuencia de frames desordenados. Por lo tanto, el número de instancias
        seleccionadas será igual a la mitad del tamaño del batch (batch_size/2)"""

        list_IDs_temp = self.list_IDs_epoch[int(index * (self.batch_size / 2)):int((index + 1) * (self.batch_size / 2))]

        #LLamo a la función para generar los datos con el identificador de las instancias que forman el batch
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __len__(self):

        if self.curriculum_learning_level == 1:
            return int(np.ceil((len(self.list_IDs) * 6) / self.batch_size))
        elif self.curriculum_learning_level == 2:
            return int(np.ceil((len(self.list_IDs) * 4) / self.batch_size))
        else:
            return int(np.ceil((len(self.list_IDs) * 2) / self.batch_size))