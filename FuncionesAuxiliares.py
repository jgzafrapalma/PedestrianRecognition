import math
import cv2
import numpy as np
import os
from os.path import isfile, join
import random
import pickle
import errno
import copy
from pathlib import Path

import argparse

from tensorflow.keras.preprocessing.image import img_to_array

import logging


# Función para extraer un número de frames por cada video del conjunto de datos

def extractFrames(pathVideos, ID_video, ped, nframes, shape=()):
    cap = cv2.VideoCapture(pathVideos + ID_video + '/' + ped + '/%3d.jpg')

    # Numero total de frames del video
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    original_total = total_frames

    if total_frames % 2 != 0:
        total_frames += 1

    frame_step = math.floor(total_frames / (nframes - 1))

    frame_step = max(1, frame_step)

    # Lista donde se van a ir almacenando aquellos frames que se van a seleccionar
    frames = []

    id_frame = 0
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        id_frame += 1
        if id_frame == 1 or id_frame % frame_step == 0 or id_frame == original_total:

            # Si se introduce un shape se redimenciona el tamaño de los frames seleccionados
            if shape:
                frame = cv2.resize(frame, (shape[1], shape[0]))

            #filename = "./datasets/frames/" + ID_video + "_" + str(id_frame) + ".jpg"

            # Se almacenan los frames del video para visualizar cuales han sido seleccionados en cada caso
            #cv2.imwrite(filename, frame)

            # Se transforma la imagen a un array
            frame = img_to_array(frame)

            # Se añade a la lista de frames
            frames.append(frame)

        if len(frames) == nframes:
            break

    cap.release()

    return np.array(frames)

def extract_Frames_Matriz(pathInstances, ID_instance, n_frames_extracted):

    #Cargamos el peaton desde el fichero de datos binario de numpy
    ped = np.load(pathInstances + '/' + ID_instance)

    #Numero total de frames en los que aparece el peaton
    total_frames = ped.shape[0]

    original_total = copy.copy(total_frames)

    #Si el número total de frames es impar se le suma 1
    if total_frames % 2 != 0:
        total_frames += 1

    #Se define el paso que se va a tomar para la resumen de los frames
    frame_step = math.floor(total_frames / (n_frames_extracted - 1))

    #Corrección para algunos casos
    frame_step = max(1, frame_step)

    # Lista donde se van a ir almacenando aquellos frames que se van a seleccionar
    frames = []

    for id_frame in range(total_frames):

        if id_frame == 0 or id_frame % frame_step == 0 or id_frame == (original_total - 1):

            frames.append(ped[id_frame])

            #if not os.path.exists('/media/jorge/DATOS/TFG/datasets/Recortes'):
                #os.mkdir('/media/jorge/DATOS/TFG/datasets/Recortes')

            #ID_instance_no_ext = ".".join(ID_instance.split(".")[:-1])

            #if not os.path.exists('/media/jorge/DATOS/TFG/datasets/Recortes' + '/' + ID_instance_no_ext):
                #os.mkdir('/media/jorge/DATOS/TFG/datasets/Recortes' + '/' + ID_instance_no_ext)

            #filename = '/media/jorge/DATOS/TFG/datasets/Recortes/' + ID_instance_no_ext + '/' + str(id_frame) + ".jpg"

            # Se almacenan los frames del video para visualizar cuales han sido seleccionados en cada caso
            #cv2.imwrite(filename, ped[id_frame])

        if len(frames) == n_frames_extracted:
            break

    return np.array(frames)


def extract_Frames_JAAD(input_frames, input_labels, n_frames_extracted, ID_video, ID_pedestrian):

    #Numero total de frames en los que aparece el peaton
    total_frames = input_frames.shape[0]

    original_total = copy.copy(total_frames)

    #Si el número total de frames es impar se le suma 1
    if total_frames % 2 != 0:
        total_frames += 1

    #Se define el paso que se va a tomar para la resumen de los frames
    frame_step = math.floor(total_frames / (n_frames_extracted - 1))

    #Corrección para algunos casos
    frame_step = max(1, frame_step)

    # Lista donde se van a ir almacenando aquellos frames que se van a seleccionar
    frames = []
    labels = []

    for id_frame in range(total_frames):

        if id_frame == 0 or id_frame % frame_step == 0 or id_frame == (original_total - 1):

            frames.append(input_frames[id_frame])

            labels.append(input_labels[id_frame])

            if not os.path.exists('/media/jorge/DATOS/TFG/datasets/Recortes'):
                os.mkdir('/media/jorge/DATOS/TFG/datasets/Recortes')

            if not os.path.exists('/media/jorge/DATOS/TFG/datasets/Recortes' + '/' + ID_video):
                os.mkdir('/media/jorge/DATOS/TFG/datasets/Recortes' + '/' + ID_video)

            if not os.path.exists('/media/jorge/DATOS/TFG/datasets/Recortes' + '/' + ID_video + '/' + ID_pedestrian):
                os.mkdir('/media/jorge/DATOS/TFG/datasets/Recortes' + '/' + ID_video + '/' + ID_pedestrian)

            filename = '/media/jorge/DATOS/TFG/datasets/Recortes' + '/' + ID_video + '/' + ID_pedestrian + '/' + str(id_frame) + ".jpg"

            # Se almacenan los frames del video para visualizar cuales han sido seleccionados en cada caso
            cv2.imwrite(filename, input_frames[id_frame])

        if len(frames) == n_frames_extracted:
            break

    return np.array(frames), np.array(labels)

def ShuffleFrames(frames, n_swaps):
    frames_Shuffle = np.ndarray(frames.shape)

    n_frames = frames.shape[0]

    indexes_start = np.zeros(n_frames)

    # Generación de los indices iniciales de los frames
    for index in range(n_frames):
        indexes_start[index] = index

    while True:

        indexes_end = np.array(indexes_start, copy=True)

        for i in range(n_frames - 1, n_frames - 1 - n_swaps, -1):
            j = random.randint(0, n_frames - 1)

            indexes_end[i], indexes_end[j] = indexes_end[j], indexes_end[i]

        if not equal_arrays(indexes_start, indexes_end):
            break

    # A partir de los indices obtenidos se obtiene la imagen final
    for i in range(n_frames):
        for j in range(n_frames):
            if indexes_end[j] == i:
                frames_Shuffle[j] = frames[i]


    return frames_Shuffle


def equal_arrays(array1, array2):

    n_elements = len(array1)

    if n_elements != len(array2):
        return False

    for i in range(len(array1)):
        if array1[i] != array2[i]:
            return False

    return True


#Función que recibe como parámetro la ruta de la carpeta donde se encuentran las instancias y los porcentajes para las
#particiones de validación y test, y genera tres ficheros con los ID de los videos pertenecientes a cada conjunto de datos
def create_train_validation_test(path_instances, percent_validation, percent_test, output_path):

    #PRECONDICIONES

    #La precondición es que los porcentajes de validación y test deben de ser un valor entre [0, 1). La suma de ambos porcentajes
    #no puede ser igual a 1, ya que sino no habria instancias en el conjunto de train
    assert 0.0 <= percent_validation < 1.0 and 0.0 <= percent_test < 1.0 and (percent_validation + percent_test) != 1.0
    #La ruta pasado como parámetro debe de ser un directorio
    assert os.path.isdir(path_instances)

    #Almaceno en una lista el nombre de los ficheros que se encuentran en la carpeta de las instancias
    files = os.listdir(path_instances)

    n_files = len(files)

    #Numero de instancias en el conjunto de validación
    n_validation = math.floor(len(files)*percent_validation)
    #Número de instancias en el conjunto de test
    n_test = math.floor(len(files)*percent_test)
    #Número de instancias en el conjunto de entrenamiento
    n_train = len(files) - n_validation - n_test

    if not n_train:
        print("Error, el número de elementos en el conjunto de entrenamiento es igual a 0.")
        return

    #Listas vacias en las que se van a almacenar los identificadores de las distintas instancias para cada subconjunto de datos
    test = []
    validation = []

    #Se rellena la lista vacia para el conjunto de entrenamiento
    for _ in range(n_test):
        #Se genera un número aleatorio entre 0 y el número de elementos de files
        index = random.randint(0, len(files) - 1)
        #Se elimina el elemento en la posición index de la lista files y es añadido a la lista del conjunto de entrenamiento
        test.append(files.pop(index))

    # Se rellena la lista vacia para el conjunto de validación
    for _ in range(n_validation):
        # Se genera un número aleatorio entre 0 y el número de elementos de files
        index = random.randint(0, len(files) - 1)
        # Se elimina el elemento en la posición index de la lista files y es añadido a la lista del conjunto de validación
        validation.append(files.pop(index))

    #Post-condición
    #La suma de elementos de train, validación y test debe de ser igual al número de elementos de la lista inicial de ficheros
    assert len(test) + len(validation) + len(files) == n_files

    #Escritura de las tres listas en un fichero .txt (para que sea visible por el usuario)

    #Se crea el directorio de salida para almacenar los ficheros en caso de que no exista
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not test == []:

        with open(output_path + '/test.txt', 'w') as filehandle:
            filehandle.writelines("%s\n" % place for place in test)

    if not validation == []:

        with open(output_path + '/validation.txt', 'w') as filehandle:
            filehandle.writelines("%s\n" % place for place in validation)

    if not files == []:

        with open(output_path + '/train.txt', 'w') as filehandle:
            filehandle.writelines("%s\n" % place for place in files)

#Función que realiza la lectura de las instancias que se encuentran en el fichero path_file y devuelve la lista con
#el nombre de las mismas
def read_instance_file_txt(path_file):

    files = []

    with path_file.open('r') as filehandle:

        files = [place.rstrip() for place in filehandle.readlines()]

    return files





#pathVideos: ruta donde se encuentran los videos
#pathInsatnces: ruta donde se quiere almacenar las instancias que se van a generar
#pathFrames: ruta donde se van a almacenar los frames del problema si la variable booleana frames esta activa

def extract_pedestrians_datasets_JAAD(pathVideos, pathInstances, pathFrames, pathData, rate, n_frames, shape=(), frames=True):

    with open(pathData, 'rb') as f:
        data = pickle.load(f)

    if frames:
        if not os.path.exists(pathFrames):
            os.mkdir(pathFrames)

    if not os.path.exists(pathInstances):
        os.mkdir(pathInstances)

    #Se recorren todos los videos
    for f in os.listdir(pathVideos):
        if isfile(join(pathVideos, f)):
            cap = cv2.VideoCapture(pathVideos + '/' + f)

            print(pathVideos + '/' + f)

            #Nombre del fichero sin extención
            f_no_ext = ".".join(f.split(".")[:-1])

            if frames:
                if not os.path.exists(pathFrames + '/' + f_no_ext):
                    os.mkdir(pathFrames + '/' + f_no_ext)

            width = data[f_no_ext]['width']
            height = data[f_no_ext]['height']

            #Lista con los peatones (personas que interactuan con el conductor) del video f_no_ext
            list_pedestrian = [ped for ped in list(data[f_no_ext]['ped_annotations']) if data[f_no_ext]['ped_annotations'][ped]['old_id'].find('pedestrian') != -1]

            #Lista donde se van a ir almacenando las matrices correspondientes a cada unos de los peatones del video f
            cuts_pedestrian = list()
            #Lista donde se van a ir almacenando las etiquetas para cada uno de los peatones del video f
            crossing_pedestrian = list()

            # Se reserva espacio para almacenar los distintos frames recortados de cada uno de los peatones
            for id_ped in list_pedestrian:
                #Numero de frames en los que aparece el peaton id_ped
                num_frames = len(data[f_no_ext]['ped_annotations'][id_ped]['frames'])
                cuts_pedestrian.append(np.zeros((num_frames, shape[0], shape[1], 3)))
                #Por cada peaton en el video se tiene un vector para almacenar la etiqueta de si el peaton cruza o no
                crossing_pedestrian.append(np.zeros(num_frames))

            id_frame = 0
            while cap.isOpened():

                ret, frame = cap.read()

                #Si no se puede abrir el video se sale del bucle while
                if not ret:
                    break

                for id_ped, ped in enumerate(list_pedestrian):

                    if frames:
                        if not os.path.exists(pathFrames + '/' + f_no_ext + '/' + ped):
                            os.mkdir(pathFrames + '/' + f_no_ext + '/' + ped)

                    #Compruebo si el peaton se encuentra en el frame actual
                    list_frames = data[f_no_ext]['ped_annotations'][ped]['frames']
                    if id_frame in list_frames:
                        #Obtengo la posición de la lista de frames del peaton en el que es encuentra en frame actual.
                        #(Va a servir para luego saber en que index consultar la bounding box)
                        index_frame = list_frames.index(id_frame)
                        #Lista con las coordendas de los dos puntos de la bounding box
                        bbox = data[f_no_ext]['ped_annotations'][ped]['bbox'][index_frame]

                        #Obtengo el valor que tiene el atributo cross para el frame index_frame (etiqueta sobre si va a cruzar o no el peatón)
                        cross = data[f_no_ext]['ped_annotations'][ped]['behavior']['cross'][index_frame]

                        diff_y = bbox[2] - bbox[0]
                        diff_x = bbox[3] - bbox[1]

                        #Si la diferencia de la coordenada de x es mayor (Caso habitual)
                        if diff_x >= diff_y:

                            #Incremento que se va a realizar sobre el recorte tanto por la parte superior como
                            #inferior de los fotogramas
                            increment = math.floor(diff_x * rate)

                            expected_size_cut = int((diff_x + 1) + 2*increment)

                            #print("Expected_size_cut: %d" % expected_size_cut)

                            #Se calcula la nueva posición que va a tener la coordenada x1
                            new_x1 = bbox[1] - increment

                            #Si pasa del marco superior de la imagen
                            if new_x1 < 0:
                                #En la imagen resultado donde se va a almacenar el recorte
                                cut_x1 = new_x1 * (-1)
                                #La nueva coordenada de x se establece en 0
                                bbox[1] = 0
                            else:
                                bbox[1] = new_x1
                                cut_x1 = 0

                            new_x2 = bbox[3] + increment

                            #Si pasa del marco inferior de la imagen
                            if new_x2 > (height - 1):
                                cut_x2 = (expected_size_cut - 1) - (new_x2 - (height - 1))
                                bbox[3] = (height - 1)
                            else:
                                bbox[3] = new_x2
                                cut_x2 = expected_size_cut - 1


                            #La longitud que va a tener la parte horizontal debera de ser igual a la nueva diferencia
                            #de las coordenadas x

                            diff = (expected_size_cut - 1) - diff_y

                            #Al ser esta diferencia un valor impar, por el lado derecho se va a incrementar en un pixel más
                            if diff % 2 != 0:
                                increment = math.floor(diff / 2)

                                new_y1 = bbox[0] - increment

                                #Se pasa del marco lateral izquierdo de la imagen
                                if new_y1 < 0:
                                    cut_y1 = new_y1 * (-1)
                                    bbox[0] = 0
                                else:
                                    bbox[0] = new_y1
                                    cut_y1 = 0


                                new_y2 = bbox[2] + increment + 1

                                if new_y2 > (width - 1):
                                    # Cantidad que me salgo hacia la derecha de la imagen
                                    cut_y2 = (expected_size_cut - 1) - (new_y2 - (width - 1))
                                    bbox[2] = (width - 1)
                                else:
                                    bbox[2] = new_y2
                                    cut_y2 = expected_size_cut - 1

                            else:

                                increment = diff / 2

                                new_y1 = bbox[0] - increment

                                # Se pasa del marco lateral izquierdo de la imagen
                                if new_y1 < 0:
                                    cut_y1 = new_y1 * (-1)
                                    bbox[0] = 0
                                else:
                                    bbox[0] = new_y1
                                    cut_y1 = 0

                                new_y2 = bbox[2] + increment

                                if new_y2 > (width - 1):
                                    cut_y2 = (expected_size_cut - 1) - (new_y2 - (width - 1))
                                    bbox[2] = (width - 1)
                                else:
                                    bbox[2] = new_y2
                                    cut_y2 = expected_size_cut - 1

                        else:

                            increment = math.floor(diff_y * rate)

                            expected_size_cut = int((diff_y + 1) + 2 * increment)

                            # Se calcula la nueva posición que va a tener la coordenada y1
                            new_y1 = bbox[0] - increment

                            # Si pasa del marco lateral izquierdo
                            if new_y1 < 0:
                                cut_y1 = new_y1 * (-1)
                                bbox[0] = 0
                            else:
                                bbox[0] = new_y1
                                cut_y1 = 0

                            # Se calcula la nueva posición que va a tener la coordenada y1
                            new_y2 = bbox[2] + increment

                            # Si pasa del marco lateral derecho
                            if new_y2 > (width - 1):
                                cut_y2 = (expected_size_cut - 1) - (new_y2 - (width - 1))
                                bbox[2] = (width - 1)
                            else:
                                bbox[2] = new_y2
                                cut_y2 = expected_size_cut - 1

                            diff = (expected_size_cut - 1) - diff_x

                            # Al ser esta diferencia un valor impar, por el lado derecho se va a incrementar en un pixel más
                            if diff % 2 != 0:
                                increment = math.floor(diff / 2)

                                new_x1 = bbox[1] - increment

                                # Se pasa del marco lateral izquierdo de la imagen
                                if new_x1 < 0:
                                    cut_x1 = new_x1 * (-1)
                                    bbox[1] = 0
                                else:
                                    bbox[1] = new_x1
                                    cut_x1 = 0

                                new_x2 = bbox[3] + increment + 1

                                if new_x2 > (height - 1):
                                    cut_x2 = (expected_size_cut - 1) - (new_x2 - (height - 1))
                                    bbox[3] = (height - 1)
                                else:
                                    bbox[3] = new_x2
                                    cut_x2 = expected_size_cut - 1
                            else:

                                increment = diff / 2

                                new_x1 = bbox[1] - increment

                                # Se pasa del marco lateral izquierdo de la imagen
                                if new_x1 < 0:
                                    cut_x1 = new_x1 * (-1)
                                    bbox[1] = 0
                                else:
                                    bbox[1] = new_x1
                                    cut_x1 = 0

                                new_x2 = bbox[3] + increment

                                if new_x2 > (height - 1):
                                    cut_x2 = (expected_size_cut - 1) - (new_x2 - (height - 1))
                                    bbox[3] = (height - 1)
                                else:
                                    bbox[3] = new_x2
                                    cut_x2 = expected_size_cut - 1

                        cut = np.zeros((expected_size_cut, expected_size_cut, 3))

                        cut[int(cut_x1):int(cut_x2+1), int(cut_y1):int(cut_y2+1)] = frame[int(bbox[1]):int(bbox[3]+1), int(bbox[0]):int(bbox[2]+1)]

                        if shape:
                            cut = cv2.resize(cut, (shape[1], shape[0]))

                        if frames:
                            cv2.imwrite(pathFrames + '/' + f_no_ext + '/' + ped + '/' + '%03d' % id_frame + '.jpg', cut)

                        cuts_pedestrian[id_ped][index_frame] = cut

                        if cross:
                            crossing_pedestrian[id_ped][index_frame] = 1

                id_frame += 1

            cap.release()

            #instance = {}

            for id_ped, cut_predestrian in enumerate(cuts_pedestrian):

                output_frames, output_labels = extract_Frames_Labels_Matriz(cut_predestrian, crossing_pedestrian[id_ped], n_frames, f_no_ext, list_pedestrian[id_ped])

                dict_ped = {'frames': output_frames, 'cross_labels': output_labels}

                # np.save(pathInstances + '/' + f_no_ext + '_' + list_pedestrian[id_ped] + '.npy', cuts_pedestrian[id_ped])

                #instance[list_pedestrian[id_ped]] = dict_ped

                with open(pathInstances + '/' + list_pedestrian[id_ped] + '.pkl', 'wb') as output:
                    pickle.dump(dict_ped, output)

            """if list_pedestrian:

                output = open(pathInstances + '/' + f_no_ext + '.pkl', 'wb')

                pickle.dump(instance, output)

                output.close()"""


def extract_pedestrian_dataset_PIE(input_path_data, input_path_dataset, output_path_instances, output_path_frames, output_path_cuts, rate, n_frames, shape=()):

    logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

    PATH_instances = Path(output_path_instances)

    PATH_frames = Path(output_path_frames)

    if not PATH_instances.exists():
        PATH_instances.mkdir()

    if not PATH_frames.exists():
        PATH_frames.mkdir()

    with open(input_path_data, 'rb') as input:
        data = pickle.load(input)

    PATH_dataset = Path(input_path_dataset)

    for set_video in PATH_dataset.iterdir():

        PATH_frames_set = Path(join(PATH_frames, set_video.name))

        logging.info("Accediendo al directorio %s" % set_video)

        #En el directorio donde se almacenan los frames se crea una carpeta para cada set
        if not PATH_frames_set.exists():
            PATH_frames_set.mkdir()

        for video in set_video.iterdir():

            if video.is_file():

                logging.info("Extrayendo peatones del video %s" % video)

                cap = cv2.VideoCapture(str(video))

                #Se crea una carpeta por cada video de cada set en la carpeta donde se almcenan los frames
                PATH_frames_video = Path(join(PATH_frames_set, video.name))

                if not PATH_frames_video.exists():
                    PATH_frames_video.mkdir()

                width = data[set_video.name][video.stem]['width']
                height = data[set_video.name][video.stem]['height']

                list_pedestrian = list(data[set_video.name][video.stem]['ped_annotations'])

                cuts_pedestrian = list()

                intention_pedestrian = list()

                crossing_pedestrian = list()

                #Se reserva memoria para almacenar los frames de cada uno de los peatones y se almacena la etiqueta de cada peaton
                for id_ped in list_pedestrian:

                    crossing = data[set_video.name][video.stem]['ped_annotations'][id_ped]['attributes']['crossing']

                    #Los casos irrelevantes son omitidos
                    if crossing == -1:
                        list_pedestrian.remove(id_ped)
                    else:
                        num_frames = len(data[set_video.name][video.stem]['ped_annotations'][id_ped]['frames'])

                        cuts_pedestrian.append(np.zeros((num_frames, shape[0], shape[1], 3)))

                        # Se rellena la lista con la probabilidad de la intencionalidad de cruzar de los distintos peatones (etiqueta a inferir)
                        """intention_pedestrian.append(
                            data[set_video.name][video.stem]['ped_annotations'][id_ped]['attributes']['intention_prob']
                        )"""
                        #Se rellena la lista con la etiqueta correspondiente si el peaton cruza la calzada o no
                        crossing_pedestrian.append(crossing)

                logging.info("Memoria para almacenar los fotogramas de los peatones recortados reservada con exito")

                id_frame = 0
                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    # Compruebo la existancia de todos los peatones en el frame id_frame
                    for id_ped, ped in enumerate(list_pedestrian):

                        PATH_frames_ped = Path(join(PATH_frames_video, ped))

                        #Se crea una carpeta por cada peaton del video donde se van a almacenar los frames
                        if not PATH_frames_ped.exists():
                            PATH_frames_ped.mkdir()

                        #Se obtiene la lista de frames del peatón ped
                        list_frames = data[set_video.name][video.stem]['ped_annotations'][ped]['frames']
                        #Se comprueba si el frame actual se encuentra en esta lista
                        if id_frame in list_frames:
                            #Se obtiene la posición del frame en la lista de frames
                            index_frame = list_frames.index(id_frame)
                            #Se obtiene el valor de la hitbox
                            bbox = data[set_video.name][video.stem]['ped_annotations'][ped]['bbox'][index_frame]

                            #Se calculan las diferencias entre las coordenadas
                            diff_x = int(bbox[2]) - int(bbox[0]) + 1
                            diff_y = int(bbox[3]) - int(bbox[1]) + 1

                            # Si la diferencia de la coordenada de x es mayor (Caso habitual)
                            if diff_x >= diff_y:

                                # Incremento que se va a realizar sobre el recorte tanto por la parte superior como
                                # inferior de los fotogramas
                                increment = math.floor(diff_x * rate)

                                expected_size_cut = int(diff_x + 2 * increment)

                                """Se calcula la nueva posición que va a tener la coordenada x1,
                                increment será igual a la cantidad de pixeles en la que se amplia la imagen por la parte
                                izquierda"""
                                new_x1 = int(bbox[0]) - increment

                                # Si pasa del marco izquierdo de la imagen
                                if new_x1 < 0:
                                    #Posición en la imagen final en la que se va a colocar el recorte
                                    cut_x1 = new_x1 * (-1)
                                    # La nueva coordenada de x se establece en 0
                                    new_x1 = 0
                                else:
                                    """La imagen no se sale por el lateral izquierdo, por lo tanto, en la imagen final 
                                    se empieza a escribir la imagen recortada desde la esquina superior izquierda"""
                                    cut_x1 = 0

                                new_x2 = int(bbox[2]) + increment

                                """Si pasa del marco derecho de la imagen (1920 o superior, ya que el ultimo valor de pixel
                                por la derecha es el 1919)"""
                                if new_x2 > (width - 1):
                                    """new_x2 - (width - 1) cantidad que se pasa por el lateral derecho,
                                        cut_x2 es la posición final en la que se recorta la imagen"""
                                    cut_x2 = (expected_size_cut - 1) - (new_x2 - (width - 1))
                                    new_x2 = (width - 1)
                                else:
                                    cut_x2 = expected_size_cut - 1

                                # La longitud que va a tener la parte vertical debera de ser igual a la nueva diferencia
                                # de las coordenadas x
                                diff = expected_size_cut - diff_y

                                # Al ser esta diferencia un valor impar, por el lado derecho se va a incrementar en un pixel más
                                if diff % 2 != 0:
                                    increment = math.floor(diff / 2)

                                    new_y1 = int(bbox[1]) - increment

                                    # Se pasa del marco superior de la imagen
                                    if new_y1 < 0:
                                        cut_y1 = new_y1 * (-1)
                                        new_y1 = 0
                                    else:
                                        cut_y1 = 0

                                    new_y2 = int(bbox[3]) + increment + 1

                                    if new_y2 > (height - 1):
                                        # Cantidad que me salgo hacia la derecha de la imagen
                                        cut_y2 = (expected_size_cut - 1) - (new_y2 - (height - 1))
                                        new_y2 = (height - 1)
                                    else:
                                        cut_y2 = expected_size_cut - 1

                                else:

                                    increment = diff / 2

                                    new_y1 = int(bbox[1]) - increment

                                    # Se pasa del marco superior de la imagen
                                    if new_y1 < 0:
                                        cut_y1 = new_y1 * (-1)
                                        new_y1 = 0
                                    else:
                                        cut_y1 = 0

                                    new_y2 = int(bbox[3]) + increment

                                    if new_y2 > (height - 1):
                                        cut_y2 = (expected_size_cut - 1) - (new_y2 - (height - 1))
                                        new_y2 = (height - 1)
                                    else:
                                        cut_y2 = expected_size_cut - 1

                            else: # diff_y > diff_x

                                increment = math.floor(diff_y * rate)

                                expected_size_cut = int(diff_y + 2 * increment)

                                # Se calcula la nueva posición que va a tener la coordenada y1
                                new_y1 = int(bbox[1]) - increment

                                # Si pasa del marco superior
                                if new_y1 < 0:
                                    cut_y1 = new_y1 * (-1)
                                    new_y1 = 0
                                else:
                                    cut_y1 = 0

                                # Se calcula la nueva posición que va a tener la coordenada y1
                                new_y2 = int(bbox[3]) + increment

                                # Si pasa del inferior
                                if new_y2 > (height - 1):
                                    cut_y2 = (expected_size_cut - 1) - (new_y2 - (height - 1))
                                    new_y2 = (height - 1)
                                else:
                                    cut_y2 = expected_size_cut - 1

                                diff = expected_size_cut - diff_x

                                # Al ser esta diferencia un valor impar, por el lado derecho se va a incrementar en un pixel más
                                if diff % 2 != 0:
                                    increment = math.floor(diff / 2)

                                    new_x1 = int(bbox[0]) - increment

                                    # Se pasa del marco lateral izquierdo de la imagen
                                    if new_x1 < 0:
                                        cut_x1 = new_x1 * (-1)
                                        new_x1 = 0
                                    else:
                                        cut_x1 = 0

                                    new_x2 = int(bbox[2]) + increment + 1

                                    if new_x2 > (width - 1):
                                        cut_x2 = (expected_size_cut - 1) - (new_x2 - (width - 1))
                                        new_x2 = (width - 1)
                                    else:
                                        cut_x2 = expected_size_cut - 1

                                else:

                                    increment = diff / 2

                                    new_x1 = int(bbox[0]) - increment

                                    # Se pasa del marco lateral izquierdo de la imagen
                                    if new_x1 < 0:
                                        cut_x1 = new_x1 * (-1)
                                        new_x1 = 0
                                    else:
                                        cut_x1 = 0

                                    new_x2 = int(bbox[2]) + increment

                                    if new_x2 > (width - 1):
                                        cut_x2 = (expected_size_cut - 1) - (new_x2 - (width - 1))
                                        new_x2 = (width - 1)
                                    else:
                                        cut_x2 = expected_size_cut - 1

                            cut = np.zeros((expected_size_cut, expected_size_cut, 3))

                            cut[int(cut_y1):int(cut_y2+1), int(cut_x1):int(cut_x2+1)] = frame[int(new_y1):int(new_y2+1), int(new_x1):int(new_x2+1)]

                            if shape:
                                cut = cv2.resize(cut, (shape[1], shape[0]))

                            cv2.imwrite(join(PATH_frames_ped, '%03d' % id_frame + '.jpg'), cut)

                            cuts_pedestrian[id_ped][index_frame] = cut

                    id_frame += 1

                cap.release()

                logging.info("Peatones del video %s recortados con exito" % video)

                for id_ped, cut_pedestrian in enumerate(cuts_pedestrian):

                    output_frames = extract_Frames_PIE(output_path_cuts, cut_pedestrian, n_frames, set_video.name, video.stem, list_pedestrian[id_ped])

                    #dict_ped = {'frames': output_frames, 'intention_prob': intention_pedestrian[id_ped], 'crossing': crossing_pedestrian[id_ped]}
                    dict_ped = {'frames': output_frames, 'crossing': crossing_pedestrian[id_ped]}

                    with open(join(output_path_instances, list_pedestrian[id_ped] + '.pkl'), 'wb') as output:
                        pickle.dump(dict_ped, output)

                logging.info("Instancias del video %s guardadas con exito" % video)


def extract_Frames_PIE(output_path_cuts, input_frames, n_frames_extracted, ID_set, ID_video, ID_pedestrian):
    # Numero total de frames en los que aparece el peaton
    total_frames = input_frames.shape[0]

    original_total = copy.copy(total_frames)

    # Si el número total de frames es impar se le suma 1
    if total_frames % 2 != 0:
        total_frames += 1

    # Se define el paso que se va a tomar para la resumen de los frames
    frame_step = math.floor(total_frames / (n_frames_extracted - 1))

    # Corrección para algunos casos
    frame_step = max(1, frame_step)

    # Lista donde se van a ir almacenando aquellos frames que se van a seleccionar
    frames = []

    #Se crea el directorio donde se van a almacenar los recortes
    PATH_cuts = Path(output_path_cuts)
    if not PATH_cuts.exists():
        PATH_cuts.mkdir()

    PATH_set_cuts = Path(join(PATH_cuts, ID_set))
    if not PATH_set_cuts.exists():
        PATH_set_cuts.mkdir()

    PATH_video_cuts = Path(join(PATH_set_cuts, ID_video))
    if not PATH_video_cuts.exists():
        PATH_video_cuts.mkdir()

    PATH_ped_cuts = Path(join(PATH_video_cuts, ID_pedestrian))
    if not PATH_ped_cuts.exists():
        PATH_ped_cuts.mkdir()

    for id_frame in range(total_frames):

        if id_frame == 0 or id_frame % frame_step == 0 or id_frame == (original_total - 1):

            frames.append(input_frames[id_frame])

            # Se almacenan los frames del video para visualizar cuales han sido seleccionados en cada caso
            cv2.imwrite(join(PATH_ped_cuts, str(id_frame) + '.jpg'), input_frames[id_frame])

        if len(frames) == n_frames_extracted:
            break

    return np.array(frames)


def save_frames(path_saves, frames, ID):

    if not os.path.exists(path_saves):
        os.mkdir(path_saves)

    for i in range(len(frames)):
        cv2.imwrite(path_saves + '/' + str(ID) + "__" + str(i) + ".jpg", frames[i])

