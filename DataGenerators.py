import numpy as np
from tensorflow.keras.utils import Sequence
from FuncionesAuxiliares import ShuffleFrames
from tensorflow.keras.utils import to_categorical

from pathlib import Path

from os.path import join
from os.path import splitext

import pickle


##########################################################################################################
###################################  PRETEXT TASK SHUFFLE  ###############################################
##########################################################################################################


class DataGeneratorShuffle(Sequence):
    def __init__(self, list_IDs, path_instances, n_frames, batch_size=32, dim=(32, 32, 32), n_channels=1, n_clases=10, shuffle=True, normalized=True, step_swaps=5):
        self.dim = dim
        self.batch_size = batch_size
        #Lista con los nombres de los peatones que forman parte del conjunto de entrenamiento
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

            with (self.path_instances / ID_instance).open('rb') as input:
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



##########################################################################################################
##################################  PRETEXT TASK ORDER PREDICTION  #######################################
##########################################################################################################


class DataGeneratorOrderPrediction(Sequence):
    def __init__(self, list_IDs, path_instances, batch_size=32, dim=(32, 32, 32), n_channels=1, n_clases=10, normalized=True, opticalFlow=False, n_epochs=100, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_clases
        self.normalized = normalized
        self.shuffle = shuffle
        #Ruta donde se encuentran las instancias
        self.path_instances = path_instances

        self.opticalFlow = opticalFlow

        #Lista con el nombre de los peatones que forman parte del conjunto de entrenamiento
        self.list_IDs = list_IDs

        #Lista con los peatones disponibles a la hora de ir rellenando los batches
        self.restant_pedestrians = self.list_IDs.copy()

        #Se crea un diccionario utilizado para ver las clases que quedan por ser seleccionadas de cada peatón
        self.restant_classes_pedestrians = dict([(ped, [0, 1, 2, 3, 4, 5])for ped in self.list_IDs])

        self.curriculum_learning_level = 1

        #Número de epocas necesarías para pasar el nivel 1 de dificultad de puzzle al nivel 2
        self.change_level1_to_level2 = n_epochs * 40
        # Número de epocas necesarías para pasar el nivel 2 de dificultad de puzzle al nivel 3
        self.change_level2_to_level3 = n_epochs * 30

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

        self.restant_pedestrians = self.list_IDs.copy()


    def __data_generation(self, index):
        #Indice en del batch en el que se van a escribir las instancias
        index_batch = 0

        #Si el indice del batch que se va a llenar es distinto del último
        if index != self.__len__():

            """Se reserva espacio para almacenar las instancias del batch actual y las etiquetas de esas instancias."""
            X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
            X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
            X3 = np.empty((self.batch_size, *self.dim, self.n_channels))
            X4 = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty(self.batch_size, dtype=int)

            #Numero de peatones que deben de ser seleccionados para rellenar el batch
            num_pedestrian = self.batch_size / 2

            #Lista en la que se van a
            selected_pedestrians = list()

            i = 0
            while i != num_pedestrian:

                # Si durante el proceso de selección la lista de peatones restantes se vacia, la reinicio con todos los peatones otra vez,
                # salvo aquellos que acaben de ser selecionados para este batch.
                if self.restant_pedestrians == []:

                    self.restant_pedestrians = self.restant_classes_pedestrians.keys()

                    #Hay que comprobar si es los peatones del batch anterior siguen siendo seleccionables para borrarlo

                    # Se eliminan de las lista de los peatones restantes aquellos que ya han sido selecionados para el ultimo batch.
                    for ped in selected_pedestrians:
                        #Si algunos de los peatones seleccionados hasta ahora sigue siendo seleccionable es eliminado
                        if ped in self.restant_pedestrians:
                            self.restant_pedestrians.remove(ped)

                #Se genera de manera aleatoria el indice para seleccionar un peatón de los restantes disponibles
                pos = np.random.randint(len(self.restant_pedestrians))

                ped = self.restant_pedestrians[pos]

                ped_noext = splitext(ped)[0]

                #Se añade a la lista de peatones seleccionados
                selected_pedestrians.append(ped)

                #Se elimina de la lista de peatones disponibles el peatón seleccionado
                self.restant_pedestrians.remove(ped)

                if self.opticalFlow:

                    PathInstance = Path(join(self.path_instances, 'OrderPrediction', 'OpticalFlow', ped_noext))

                else:

                    PathInstance = Path(join(self.path_instances, 'OrderPrediction', 'NotOpticalFlow', ped_noext))


                #Se obtienen las clases disponibles del peatón seleccionado
                classes = self.restant_classes_pedestrians[ped]

                #Del peatón seleccionado se selecciona de manera aleatoria una de sus clases
                pos_class = np.random.randint(len(classes))

                clas = classes[pos_class]

                classes.remove(clas)

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
                pos_class = np.random.randint(len(classes))

                clas = classes[pos_class]

                classes.remove(clas)

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

                #HAY QUE COMPROBAR SI AL ELIMINAR LAS CLASES ESTAS SON IGUAL A CERO, EN ESE CASO EL PEATÓN QUEDA ELIMINADO DE PODER SER ELEGIDO PERMANENTEMENTE

                if self.restant_classes_pedestrians[ped] == []:
                    _ = self.restant_classes_pedestrians.pop(ped)

                i += 1

        #En este caso dependiendo el número de peatones que se tenga en el instante inicial se podrá llenar completamente el batch o no
        else:

            batch_size = len(self.restant_classes_pedestrians.keys())*2

            """Se reserva espacio para almacenar las instancias del batch actual y las etiquetas de esas instancias."""
            X1 = np.empty((batch_size, *self.dim, self.n_channels))
            X2 = np.empty((batch_size, *self.dim, self.n_channels))
            X3 = np.empty((batch_size, *self.dim, self.n_channels))
            X4 = np.empty((batch_size, *self.dim, self.n_channels))
            y = np.empty(batch_size, dtype=int)

            num_pedestrian = batch_size / 2

            i = 0

            self.restant_pedestrians = self.restant_classes_pedestrians.keys()

            while i != num_pedestrian:

                #Se genera de manera aleatoria el indice para seleccionar un peatón de los restantes disponibles
                pos = np.random.randint(len(self.restant_pedestrians))

                ped = self.restant_pedestrians[pos]

                ped_noext = splitext(ped)[0]

                #Se elimina de la lista de peatones disponibles el peatón seleccionado
                self.restant_pedestrians.remove(ped)

                if self.opticalFlow:

                    PathInstance = Path(join(self.path_instances, 'OrderPrediction', 'OpticalFlow', ped_noext))

                else:

                    PathInstance = Path(join(self.path_instances, 'OrderPrediction', 'NotOpticalFlow', ped_noext))


                #Se obtienen las clases disponibles del peatón seleccionado
                classes = self.restant_classes_pedestrians[ped]

                #Del peatón seleccionado se selecciona de manera aleatoria una de sus clases
                pos_class = np.random.randint(len(classes))

                clas = classes[pos_class]

                classes.remove(clas)

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
                pos_class = np.random.randint(len(classes))

                clas = classes[pos_class]

                classes.remove(clas)

                with (PathInstance / (ped_noext + '_' + str(clas) + 'p.pkl')).open('rb') as input:
                    instance2 = pickle.load(input)

                frames = instance2['frames']

                # Normalización de los frames
                if self.normalized:
                    frames = frames * 1 / 255

                X1[index_batch, ] = frames[0]
                X2[index_batch, ] = frames[1]
                X3[index_batch, ] = frames[2]
                X4[index_batch, ] = frames[3]

                y[index_batch, ] = instance2['class']

                index_batch += 1

                #HAY QUE COMPROBAR SI AL ELIMINAR LAS CLASES ESTAS SON IGUAL A CERO, EN ESE CASO EL PEATÓN QUEDA ELIMINADO DE PODER SER ELEGIDO PERMANENTEMENTE

                if self.restant_classes_pedestrians[ped] == []:
                    _ = self.restant_classes_pedestrians.pop(ped)

                i += 1

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

        #LLamo a la función para generar los datos con el identificador de las instancias que forman el batch
        X, y = self.__data_generation(index)

        return X, y

    def __len__(self):

        if self.curriculum_learning_level == 1:
            return int(np.floor(len(self.list_IDs) * 6 / self.batch_size))
        elif self.curriculum_learning_level == 2:
            return int(np.floor(len(self.list_IDs) * 4 / self.batch_size))
        else:
            return int(np.floor(len(self.list_IDs) * 2 / self.batch_size))



##########################################################################################################
###########################################  FINAL MODELS  ###############################################
##########################################################################################################



class DataGeneratorFINALCrossingDetection(Sequence):
    def __init__(self, list_IDs, path_instances, n_frames, batch_size=32, dim=(128, 128, 32), n_channels=1, n_clases=1, shuffle=True, normalized=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_clases
        self.shuffle = shuffle
        self.n_frames = n_frames
        self.normalized = normalized
        self.path_instances = path_instances
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))  
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, self.n_frames, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=float)

        for i, ID_instance in enumerate(list_IDs_temp):

            with (self.path_instances / ID_instance).open('rb') as input:
                instance = pickle.load(input)

            """Me quedo unicamente con el campo de frames de la instancia que almacena el vector de numpy con los fotograma
            de los vídeos"""
            frames = instance['frames']

            label = instance['crossing']

            #Normalización de los frames
            if self.normalized:
                frames = frames * 1 / 255

            #Se almacenan los frames ordenados y su etiqueta
            X[i, ] = frames
            y[i] = label

        return X, to_categorical(y, num_classes=self.n_classes)

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
        return int(np.floor(len(self.list_IDs) / self.batch_size))



"""class DataGeneratorFINALRegression(Sequence):
    def __init__(self, list_IDs, path_instances, n_frames, batch_size=32, dim=(128, 128, 32), n_channels=1, n_clases=1, shuffle=True, normalized=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_clases = n_clases
        self.shuffle = shuffle
        self.n_frames = n_frames
        self.normalized = normalized
        self.path_instances = path_instances
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))  
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, self.n_frames, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=float)

        for i, ID_instance in enumerate(list_IDs_temp):

            with (self.path_instances / ID_instance).open('rb') as input:
                instance = pickle.load(input)

            Me quedo unicamente con el campo de frames de la instancia que almacena el vector de numpy con los fotograma
            de los vídeos
            frames = instance["frames"]

            intention_prob = instance["intention_prob"]

            #Normalización de los frames
            if self.normalized:
                frames = frames * 1 / 255

            #Se almacenan los frames ordenados y su etiqueta
            X[i, ] = frames
            y[i] = intention_prob


        return X, y

    def __getitem__(self, index):

        indexes_batch = self.indexes[int(index * self.batch_size):int((index + 1) * self.batch_size)]

        #Obtengo el identificador de las instancias que van a estar en el batch index
        list_IDs_temp = [self.list_IDs[k] for k in indexes_batch]

        #LLamo a la función para generar los datos con el identificador de las instancias que forman el batch
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def get_ID_instances_and_real_labels(self):

        #Obtengo el identificador de las instancias en el orden en el que son generadas
        ID_instances = [self.list_IDs[k] for k in self.indexes]

        real_labels = []
        for ID_instance in ID_instances:

            with (self.path_instances / ID_instance).open('rb') as input:
                instance = pickle.load(input)

            real_labels.append(instance["intention_prob"])

        return ID_instances, real_labels

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

class DataGeneratorFINALClassification(Sequence):
    def __init__(self, list_IDs, path_instances, n_frames, batch_size=32, dim=(128, 128, 32), n_channels=1, n_clases=1, shuffle=True, normalized=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_clases = n_clases
        self.shuffle = shuffle
        self.n_frames = n_frames
        self.normalized = normalized
        self.path_instances = path_instances
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))  
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, self.n_frames, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=float)

        for i, ID_instance in enumerate(list_IDs_temp):

            with (self.path_instances / ID_instance).open('rb') as input:
                instance = pickle.load(input)

            Me quedo unicamente con el campo de frames de la instancia que almacena el vector de numpy con los fotograma
            de los vídeos
            frames = instance["frames"]

            intention_prob = instance["intention_prob"]

            if intention_prob >= 0.5:
                y[i] = 1
            else:
                y[i] = 0

            #Normalización de los frames
            if self.normalized:
                frames = frames * 1 / 255

            #Se almacenan los frames ordenados y su etiqueta
            X[i, ] = frames

        return X, y

    def __getitem__(self, index):

        indexes_batch = self.indexes[int(index * self.batch_size):int((index + 1) * self.batch_size)]

        #Obtengo el identificador de las instancias que van a estar en el batch index
        list_IDs_temp = [self.list_IDs[k] for k in indexes_batch]

        #LLamo a la función para generar los datos con el identificador de las instancias que forman el batch
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def get_ID_instances_and_real_labels(self):

        #Obtengo el identificador de las instancias en el orden en el que son generadas
        ID_instances = [self.list_IDs[k] for k in self.indexes]

        real_labels = []
        for ID_instance in ID_instances:

            with (self.path_instances / ID_instance).open('rb') as input:
                instance = pickle.load(input)

            if instance["intention_prob"] >= 0.8:
                real_labels.append(1)
            else:
                real_labels.append(0)

        return ID_instances, real_labels

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))"""

    