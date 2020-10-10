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

from sklearn.model_selection import train_test_split

import argparse

from tensorflow.keras.preprocessing.image import img_to_array

import logging


"""def extract_Frames_Matriz(pathInstances, ID_instance, n_frames_extracted):

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

    return np.array(frames)"""


"""def extract_Frames_JAAD(input_frames, input_labels, n_frames_extracted, ID_video, ID_pedestrian):

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

    return np.array(frames), np.array(labels)"""



########################################################################################################################
##########################################  TRAIN, VALIDATION, TEST ####################################################
########################################################################################################################



"""Función que recibe como parámetro un diccionario en el que se encuentra la información de las instancias del problema
junto con su clase, y se realiza una separación de manera estratificada en conjuntos de datos de train, validación y test"""
def create_train_validation_test(path_instances, percent_validation, percent_test, path_output):

    #PRECONDICION

    #La precondición es que los porcentajes de validación y test deben de ser un valor entre [0, 1). La suma de ambos porcentajes
    #no puede ser igual a 1, ya que sino no habria instancias en el conjunto de train
    assert 0.0 <= percent_validation < 1.0 and 0.0 <= percent_test < 1.0 and (percent_validation + percent_test) != 1.0

    #LECTURA DE LAS INSTANCIAS
    Path_Instances = Path(path_instances)

    with Path_Instances.open('rb') as file_descriptor:
        data = pickle.load(file_descriptor)

    #Almaceno en una lista el nombre de los ficheros que se encuentran en la carpeta de las instancias
    X = list(data.keys())
    y = list(data.values())

    percent_validation_and_test = percent_validation+percent_test

    #Se obtiene el conjunto de entrenamiento
    X_train, X_validation_and_test, y_train, y_validation_and_test = train_test_split(X, y, test_size=percent_validation_and_test, stratify=y)

    #Se obtienen los conjuntos de validación y de test
    X_validation, X_test, y_validation, y_test = train_test_split(X_validation_and_test, y_validation_and_test, test_size=percent_test/percent_validation_and_test, stratify=y_validation_and_test)

    #Post-condición
    #La suma de elementos de train, validación y test debe de ser igual al número de elementos de la lista inicial de ficheros
    assert len(X_test) + len(X_validation) + len(X_train) == len(X)

    #Escritura de las tres listas en un fichero .txt (para que sea visible por el usuario)

    Path_Output = Path(path_output)

    #Se crea el directorio de salida para almacenar los ficheros en caso de que no exista
    Path_Output.mkdir(parents=True, exist_ok=True)

    with (Path_Output / 'test.txt').open('w') as file_descriptor:
        file_descriptor.writelines("%s\n" % place for place in X_test)

    with (Path_Output / 'validation.txt').open('w') as file_descriptor:
        file_descriptor.writelines("%s\n" % place for place in X_validation)

    with (Path_Output / 'train.txt').open('w') as file_descriptor:
        file_descriptor.writelines("%s\n" % place for place in X_train)

#Función que realiza la lectura de las instancias que se encuentran en el fichero path_file y devuelve la lista con
#el nombre de las mismas
def read_instance_file_txt(path_file):

    files = []

    with path_file.open('r') as filehandle:

        files = [place.rstrip() for place in filehandle.readlines()]

    return files



########################################################################################################################
##########################################  FUNCIONES AUXILIARES  ######################################################
########################################################################################################################



# Función para extraer un número de frames por cada video del conjunto de datos
def extractFrames(pathFrames, nframes):

    cap = cv2.VideoCapture(str(pathFrames))

    # Numero total de frames del video
    #total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1

    # Se obtiene el ancho y alto de los fotogramas
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #original_total = total_frames

    #if total_frames % 2 != 0:
        #total_frames += 1

    #frame_step = math.floor(total_frames / (nframes - 1))

    #frame_step = max(1, frame_step)

    #Vector en el que se encuentra la posición los frames que se van a extraer del video, más dos posiciones extra
    indexes = np.linspace(start=0, stop=total_frames, num=(nframes + 2), endpoint=True, dtype=int)

    #Se elimina la última y la primera posición del vector de indices.
    indexes = np.delete(indexes, len(indexes)-1)
    indexes = np.delete(indexes, 0)


    # Se reserva memoria para almacenar los frames que van a formar parte de la instancia
    output_frames = np.zeros((nframes, width, height, 3))
    #Posición en la que se va a escribir en el vector que almacena los fotogramas extraidos
    index = 0

    id_frame = 0
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Se comprueba si el índice del frame actual se encuentra en el vector de indices que van a ser seleccionados
        if id_frame in indexes:
            #En caso afirmativo el frame es almacenado
            output_frames[index] = frame
            index += 1

        """id_frame += 1
        if id_frame == 1 or id_frame % frame_step == 0 or id_frame == original_total - 10:

            output_frames[index] = frame
            index += 1"""
        #Una vez que indice es igual al número de frames que se quieren seleccionar se termina el proceso
        if index == nframes:
            break

    cap.release()

    return output_frames


# Función para extraer un número de frames por cada video del conjunto de datos
def extractFramesOpticalFlow(pathFrames, nframes):

    cap = cv2.VideoCapture(str(pathFrames))

    # Número de frames de los que se van a calcular el flujo optico(número total de frames en los que aparece el peatón menos 1)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # Vector para almacenar las magnitudes de los flujos opticos
    magnitudes = np.zeros(num_frames)

    # Se obtiene el ancho y alto de los fotogramas
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Se reserva memoria para almacenar los fotogramas del peatón en los que se va a calcular el flujo optico
    # Se almacenan los frames una vez que se van leyendo para luego ahorrar lecturas en disco
    frames = np.zeros((num_frames, width, height, 3))

    # Se lee el primer frame del peatón (este frame no será almacenado, la magnitud del flujo óptico es calculado del segundo frame en adelante
    ret, first_frame = cap.read()

    # Se calcula el frame actual pero en escala de grises
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    index_frame = 0
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Se almacenan los frames del peaton
        frames[index_frame] = frame

        # Se convierte el frame actual a blanco y negro (necesario para calcular el flujo optico)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Se calcula el flujo optico
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Se calcula la magnitud del flujo óptico
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Se almacena la mediana de la magnitud del flujo óptico (para evitar outliers)
        magnitudes[index_frame] = np.median(magnitude)

        # Se actualiza el frame anterior para la siguiente iteración del bucle
        prev_gray = gray

        index_frame += 1

    cap.release()

    # Se obtienen los indeces de los frames con un valor de magnitud del flujo de movimiento mayor
    indexes = (-magnitudes).argsort()[:nframes]

    # Se ordenan los indices obtenidos para hacer el recorte de los frames por orden
    sort_indexes = np.sort(indexes)

    # Se reserva memoria para almacenar los frames que van a formar parte de la instancia
    output_frames = np.zeros((nframes, width, height, 3))

    # Lectura de aquellos frames que van a formar parte de la instancia (mayor flujo óptico)
    for id_out, index in enumerate(sort_indexes):
        # Se almacenan los frames que van a estar en la instancia
        output_frames[id_out] = frames[index]

    return output_frames



def cut_reshape_frame_pedestrian(width, height, rate, frame, bbox, shape):

    diff_x = int(bbox[2]) - int(bbox[0]) + 1
    diff_y = int(bbox[3]) - int(bbox[1]) + 1

    # Si la diferencia de la coordenada de x es mayor (Caso habitual)
    if diff_x >= diff_y:

        # Incremento que se va a realizar sobre el recorte tanto por la parte superior como
        # inferior de los fotogramas
        increment = math.floor(diff_x * rate)

        expected_size_cut = int(diff_x + 2 * increment)

        # Se calcula la nueva posición que va a tener la coordenada x1
        new_x1 = int(bbox[0]) - increment

        # Si pasa del marco superior de la imagen
        if new_x1 < 0:
            # En la imagen resultado donde se va a almacenar el recorte
            cut_x1 = new_x1 * (-1)
            # La nueva coordenada de x se establece en 0
            new_x1 = 0
        else:
            cut_x1 = 0

        new_x2 = int(bbox[2]) + increment

        # Si pasa del marco inferior de la imagen
        if new_x2 > (width - 1):
            cut_x2 = (expected_size_cut - 1) - (new_x2 - (width - 1))
            new_x2 = (width - 1)
        else:
            cut_x2 = expected_size_cut - 1

        # La longitud que va a tener la parte horizontal debera de ser igual a la nueva diferencia
        # de las coordenadas x

        diff = expected_size_cut - diff_y

        # Al ser esta diferencia un valor impar, por el lado derecho se va a incrementar en un pixel más
        if diff % 2 != 0:
            increment = math.floor(diff / 2)

            new_y1 = int(bbox[1]) - increment

            # Se pasa del marco lateral izquierdo de la imagen
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

            # Se pasa del marco lateral izquierdo de la imagen
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

    else:

        increment = math.floor(diff_y * rate)

        expected_size_cut = int(diff_y + 2 * increment)

        # Se calcula la nueva posición que va a tener la coordenada y1
        new_y1 = int(bbox[1]) - increment

        # Si pasa del marco lateral izquierdo
        if new_y1 < 0:
            cut_y1 = new_y1 * (-1)
            new_y1 = 0
        else:
            cut_y1 = 0

        # Se calcula la nueva posición que va a tener la coordenada y1
        new_y2 = int(bbox[3]) + increment

        # Si pasa del marco lateral derecho
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

    cut[int(cut_y1):int(cut_y2 + 1), int(cut_x1):int(cut_x2 + 1)] = frame[int(new_y1):int(new_y2 + 1), int(new_x1):int(new_x2 + 1)]

    if shape:
        cut = cv2.resize(cut, (shape[1], shape[0]))

    return cut


########################################################################################################################
############################################  JAAD DATASET  ############################################################
########################################################################################################################


#input_path_data: ruta donde se encuentra la información sobre el conjunto de datos (bboxing, labels, ...)
#input_path_dataset: ruta donde se encuentran los videos
#output_pathFrames: ruta donde se van a almacenar los frames del video
#pathFrames: ruta donde se van a almacenar los frames del problema si la variable booleana frames esta activa
def extract_pedestriansFrames_datasets_JAAD(input_path_data, input_path_dataset, output_path_frames, rate, shape=()):

    #Se abre el fichero de datos
    with open(input_path_data, 'rb') as f:
        data = pickle.load(f)

    path_Videos = Path(input_path_dataset)

    #Se recorren todos los videos
    for video in path_Videos.iterdir():

        if video.is_file():

            cap = cv2.VideoCapture(str(video))

            print(video)

            width = data[video.stem]['width']
            height = data[video.stem]['height']

            #Lista con los peatones (personas que interactuan con el conductor) del video f_no_ext
            list_pedestrian = [ped for ped in list(data[video.stem]['ped_annotations']) if data[video.stem]['ped_annotations'][ped]['old_id'].find('pedestrian') != -1]

            #Lista para llevar la cuenta del número de frames por cada peatón
            indexes_frames_pedestrian = np.zeros(len(list_pedestrian))

            #Lista donde se va a almacenar la ruta donde se van a almacenar los frames de cada peatón del video
            list_path_Frames = list()

            #Lista donde se van a ir almacenando las matrices correspondientes a cada unos de los peatones del video f
            #cuts_pedestrian = list()
            #Lista donde se van a ir almacenando las etiquetas para cada uno de los peatones del video f
            #crossing_pedestrian = list()

            # Se crean los directorios para almacenar los frames
            for id_ped in list_pedestrian:
                #Numero de frames en los que aparece el peaton id_ped
                #num_frames = len(data[video.stem]['ped_annotations'][id_ped]['frames'])
                #cuts_pedestrian.append(np.zeros((num_frames, shape[0], shape[1], 3)))
                #Por cada peaton en el video se tiene un vector para almacenar la etiqueta de si el peaton cruza o no
                #crossing_pedestrian.append(np.zeros(num_frames))

                #Se crean los directorios para almacenar los frames de los peatones de video.stem
                path_Frames = Path(join(output_path_frames, video.stem, id_ped))
                list_path_Frames.append(path_Frames)
                path_Frames.mkdir(parents=True, exist_ok=True)

            id_frame = 0
            while cap.isOpened():

                ret, frame = cap.read()

                #Si no se puede abrir el video se sale del bucle while
                if not ret:
                    break

                #Para el frame actual, recorto todos los peatones que se encuentren en el
                for id_ped, ped in enumerate(list_pedestrian):

                    #Obtengo los frames del video en los que se encuentra ped
                    list_frames = data[video.stem]['ped_annotations'][ped]['frames']
                    #Compruebo si el peaton se encuentra en el frame actual del video
                    if id_frame in list_frames:
                        #Obtengo la posición de la lista de frames del peaton en el que es encuentra en frame actual.
                        #(Va a servir para luego saber en que index consultar la bounding box)
                        index_frame = list_frames.index(id_frame)
                        #Lista con las coordendas de los dos puntos de la bounding box
                        bbox = data[video.stem]['ped_annotations'][ped]['bbox'][index_frame]

                        #Se obtiene el recorte en el frame actual para el peaton id_ped
                        cut = cut_reshape_frame_pedestrian(width, height, rate, frame, bbox, shape)

                        #Se almacena el recorte en la ruta correspondente a id_ped
                        cv2.imwrite(str(list_path_Frames[id_ped] / ('%03d' % indexes_frames_pedestrian[id_ped] + '.jpg')), cut)
                        indexes_frames_pedestrian[id_ped] = indexes_frames_pedestrian[id_ped] + 1

                        #cuts_pedestrian[id_ped][index_frame] = cut

                        #if cross:
                            #crossing_pedestrian[id_ped][index_frame] = 1

                id_frame += 1

            cap.release()



def create_instances_JAAD(input_path_data, input_path_frames, output_path_cuts, output_path_instances, n_frames):

    #Se abre el fichero de datos
    with open(input_path_data, 'rb') as file_descriptor:
        data = pickle.load(file_descriptor)

    #Ruta donde se encuentran los frames de cada peaton por video
    Path_Frames = Path(input_path_frames)

    #Se recorren todos los videos
    for video in Path_Frames.iterdir():

        print(video)

        #Se recorren los peatones por cada video
        for ped in video.iterdir():

            #Número de frames de los que se van a calcular el flujo optico(número total de frames en los que aparece el peatón menos 1)
            num_frames = len(data[video.stem]['ped_annotations'][ped.stem]['frames']) - 1

            #Se obtiene la etiqueta de si el peatón esta cruzando o no
            crossing = data[video.stem]['ped_annotations'][ped.stem]['attributes']['crossing']

            #Vector para almacenar las magnitudes de los flujos opticos
            magnitudes = np.zeros(num_frames)

            #Se inicializa la lectura de los fotogramas del peatón
            cap = cv2.VideoCapture(join(ped, '%03d.jpg'))

            #Se obtiene el ancho y alto de los fotogramas
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            #Se reserva memoria para almacenar los fotogramas del peatón en los que se va a calcular el flujo optico
            #Se almacenan los frames una vez que se van leyendo para luego ahorrar lecturas en disco
            frames = np.zeros((num_frames, width, height, 3))

            #Se lee el primer frame del peatón (este frame no será almacenado, la magnitud del flujo óptico es calculado del segundo frame en adelante
            ret, first_frame = cap.read()

            #Se calcula el frame actual pero en escala de grises
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

            index_frame = 0
            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                #Se almacenan los frames del peaton
                frames[index_frame] = frame

                #Se convierte el frame actual a blanco y negro (necesario para calcular el flujo optico)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #Se calcula el flujo optico
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                #Se calcula la magnitud del flujo óptico
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                #Se almacena la mediana de la magnitud del flujo óptico (para evitar outliers)
                magnitudes[index_frame] = np.median(magnitude)

                #Se actualiza el frame anterior para la siguiente iteración del bucle
                prev_gray = gray

                index_frame += 1

            cap.release()

            #Se obtienen los indeces de los frames con un valor de magnitud del flujo de movimiento mayor
            indexes = (-magnitudes).argsort()[:n_frames]

            #Se ordenan los indices obtenidos para hacer el recorte de los frames por orden
            sort_indexes = np.sort(indexes)

            #Se crea la carpeta para almacenar los recortes
            Path_Cuts = Path(join(output_path_cuts, str(crossing), ped.stem))
            Path_Cuts.mkdir(parents=True, exist_ok=True)

            #Se reserva memoria para almacenar los frames que van a formar parte de la instancia
            output_frames = np.zeros((n_frames, width, height, 3))

            #Lectura de aquellos frames que van a formar parte de la instancia (mayor flujo óptico)
            for id_out, index in enumerate(sort_indexes):
                #Se almacenan los frames que van a estar en la instancia
                output_frames[id_out] = frames[index]
                #Se almacenan los recortes
                cv2.imwrite(str(Path_Cuts / ('%03d' % (index + 1) + '.jpg')), frames[index])

            #Se crea la instancia
            instance = {'frames': output_frames, 'crossing': crossing}

            Path_Instances = Path(join(output_path_instances, str(crossing)))
            Path_Instances.mkdir(parents=True, exist_ok=True)

            Path_Instances = Path_Instances / (ped.stem + '.pkl')

            with Path_Instances.open('wb') as file_descriptor:
                pickle.dump(instance, file_descriptor)


########################################################################################################################
############################################  PIE DATASET  #############################################################
########################################################################################################################



"""Función que permite convertir los clips en los que aparecen los peatones en el conjunto de datos original en fotogramas,
recortando unicamente la región en la que se encuentra el peatón (bounding box)"""
def extract_pedestriansFrames_datasets_PIE(input_path_data, input_path_dataset, output_path_frames, rate, shape=()):

    logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

    #Se abre el fichero de datos
    with open(input_path_data, 'rb') as file_descriptor:
        data = pickle.load(file_descriptor)

    PATH_dataset = Path(input_path_dataset)

    for set_video in PATH_dataset.iterdir():

        logging.info("Accediendo al directorio %s" % set_video)

        for video in set_video.iterdir():

            if video.is_file():

                logging.info("Extrayendo peatones del video %s" % video)

                cap = cv2.VideoCapture(str(video))

                width = data[set_video.name][video.stem]['width']
                height = data[set_video.name][video.stem]['height']
                
                #list_pedestrian = [ped for ped in list(data[set_video.name][video.stem]['ped_annotations']) if data[set_video.name][video.stem]['ped_annotations'][ped]['attributes']['crossing'] != -1]

                list_pedestrian = list(data[set_video.name][video.stem]['ped_annotations'])

                indexes_frames_pedestrian = np.zeros(len(list_pedestrian))

                # Lista donde se va a almacenar la ruta donde se van a almacenar los frames de cada peatón del video
                list_path_Frames = list()

                #Se reserva memoria para almacenar los frames de cada uno de los peatones y se almacena la etiqueta de cada peaton
                for id_ped in list_pedestrian:

                    path_Frames = Path(join(output_path_frames, set_video.stem, video.stem, id_ped))
                    list_path_Frames.append(path_Frames)
                    path_Frames.mkdir(parents=True, exist_ok=True)

                id_frame = 0
                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    # Compruebo la existancia de todos los peatones en el frame id_frame
                    for id_ped, ped in enumerate(list_pedestrian):

                        #Se obtiene la lista de frames del peatón ped
                        list_frames = data[set_video.name][video.stem]['ped_annotations'][ped]['frames']
                        #Se comprueba si el frame actual se encuentra en esta lista
                        if id_frame in list_frames:
                            #Se obtiene la posición del frame en la lista de frames
                            index_frame = list_frames.index(id_frame)
                            #Se obtiene el valor de la hitbox
                            bbox = data[set_video.name][video.stem]['ped_annotations'][ped]['bbox'][index_frame]

                            cut = cut_reshape_frame_pedestrian(width, height, rate, frame, bbox, shape)

                            cv2.imwrite(str(list_path_Frames[id_ped] / ('%04d.jpg' % indexes_frames_pedestrian[id_ped])), cut)
                            indexes_frames_pedestrian[id_ped] = indexes_frames_pedestrian[id_ped] + 1

                    id_frame += 1

                cap.release()

                logging.info("Peatones del video %s recortados con exito" % video)



#Función que permite crear instancias que van a ser utilizadas para la evaluación del modelo final
def create_instances_PIE(input_path_data, input_path_frames, output_path_cuts, output_path_instances, n_frames, optical_flow=False):

    logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

    # Se abre el fichero de datos
    with open(input_path_data, 'rb') as file_descriptor:
        data = pickle.load(file_descriptor)

    # Ruta donde se encuentran los frames de cada peaton por video
    Path_Frames = Path(input_path_frames)

    for set_video in Path_Frames.iterdir():

        logging.info("Accediendo al directorio %s" % set_video)

        for video in set_video.iterdir():

            logging.info("Creando instancias del video %s" % video)

            for ped in video.iterdir():

                Path_Ped = Path(join(ped, '%04d.jpg'))

                logging.info("Creando instancia %s" % Path_Instance.stem)

                #Se obtiene la etiqueta de si el peatón esta cruzando o no
                crossing = data[set_video.stem][video.stem]['ped_annotations'][ped.stem]['attributes']['crossing']

                #Solamente se va a calcular para aquellos peatones que estan cruzando o no
                if crossing != -1:

                    if optical_flow:

                        Path_Instances = Path(join(output_path_instances, 'Crossing-Detection', str(n_frames) + '_frames', 'OpticalFlow'))

                        Path_Cuts = Path(join(output_path_cuts, 'Crossing-Detection', str(n_frames) + '_frames', 'OpticalFlow', ped.stem))

                        Path_Instances.mkdir(parents=True, exist_ok=True)

                        Path_Cuts.mkdir(parents=True, exist_ok=True)

                        # Comprobar si ya existe la instancia
                        Path_Instance = Path_Instances / (ped.stem + '.pkl')

                        if not Path_Instance.exists():

                            output_frames = extractFramesOpticalFlow(Path_Ped, n_frames)

                            # Se crea la instancia
                            instance = {'frames': output_frames, 'crossing': crossing}

                            with Path_Instance.open('wb') as file_descriptor:
                                pickle.dump(instance, file_descriptor)

                            for index, frame in enumerate(output_frames):
                                cv2.imwrite(str(Path_Cuts / ('%02d' % (index) + '.jpg')), frame)

                            logging.info("Instancia %s creada con exito" % Path_Instance.stem)

                        else:

                            logging.info("La instancia %s no se ha creado porque ya existe" % Path_Instance.stem)

                    else:

                        Path_Instances = Path(join(output_path_instances, 'Crossing-Detection', str(n_frames) + '_frames', 'Distributed'))

                        Path_Cuts = Path(join(output_path_cuts, 'Crossing-Detection', str(n_frames) + '_frames', 'Distributed', ped.stem))

                        Path_Instances.mkdir(parents=True, exist_ok=True)

                        Path_Cuts.mkdir(parents=True, exist_ok=True)

                        # Comprobar si ya existe la instancia
                        Path_Instance = Path_Instances / (ped.stem + '.pkl')

                        if not Path_Instance.exists():

                            output_frames = extractFrames(Path_Ped, n_frames)

                            # Se crea la instancia
                            instance = {'frames': output_frames, 'crossing': crossing}

                            with Path_Instance.open('wb') as file_descriptor:
                                pickle.dump(instance, file_descriptor)

                            for index, frame in enumerate(output_frames):
                                cv2.imwrite(str(Path_Cuts / ('%02d' % (index) + '.jpg')), frame)

                            logging.info("Instancia %s creada con exito" % Path_Instance.stem)

                        else:

                            logging.info("La instancia %s no se ha creado porque ya existe" % Path_Instance.stem)



########################################################################################################################
#############################################  PRETEXT TASK SHUFFLE  ###################################################
########################################################################################################################



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



########################################################################################################################
##########################################  PRETEXT TASK ORDER PREDICTION  #############################################
########################################################################################################################



def create_instances_PIE_OrderPrediction(input_path_frames, output_path_cuts, output_path_instances, optical_flow):

    logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

    # Ruta donde se encuentran los frames de cada peaton por video
    Path_Frames = Path(input_path_frames)

    for set_video in Path_Frames.iterdir():

        logging.info("Accediendo al directorio %s" % set_video)

        for video in set_video.iterdir():

            logging.info("Creando instancias del video %s" % video)

            for ped in video.iterdir():

                Path_Ped = Path(join(ped, '%04d.jpg'))

                #logging.info("Creando instancia %s" % Path_Instance.stem)

                if optical_flow:

                    Path_Instances = Path(join(output_path_instances, 'OrderPrediction', 'OpticalFlow', ped.stem))

                    Path_Cuts = Path(join(output_path_cuts, 'OrderPrediction', 'OpticalFlow', ped.stem))

                    output_frames = extractFramesOpticalFlow(Path_Ped, 4)

                else:

                    Path_Instances = Path(join(output_path_instances, 'OrderPrediction', 'Distributed', ped.stem))

                    Path_Cuts = Path(join(output_path_cuts, 'OrderPrediction', 'Distributed', ped.stem))

                    output_frames = extractFrames(Path_Ped, 4)

                Path_Instances.mkdir(parents=True, exist_ok=True)

                Path_Cuts.mkdir(parents=True, exist_ok=True)

                create_permutations_OrderPrediction(Path_Instances, Path_Cuts, ped, output_frames)

                logging.info("Permutaciones(instancias) para el peatón %s creadas con exitos" % ped.stem)



def create_permutations_OrderPrediction(Path_Instances, Path_Cuts, ped, output_frames):

    # CALCULO DE LAS PERMUTACIONES

    # Clase 0 (vector sin permutación) {a, b, c, d}

    Path_Instance = Path_Instances / (ped.stem + '_0p.pkl')

    instance = {'frames': output_frames, 'class': 0}

    # Se almacena la instancia correspondiente a la primera permutación
    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación

    Path_Cut = Path_Cuts / '0p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(output_frames):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 1 (una permutación) {a, b, d, c}

    Path_Instance = Path_Instances / (ped.stem + '_1p.pkl')

    permutation_1 = permutation_vector(output_frames, 2, 3)

    instance = {'frames': permutation_1, 'class': 1}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación

    Path_Cut = Path_Cuts / '1p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_1):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 2 (una permutación) {a, d, c, b}

    Path_Instance = Path_Instances / (ped.stem + '_2p.pkl')

    permutation_2 = permutation_vector(output_frames, 1, 3)

    instance = {'frames': permutation_2, 'class': 2}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación
    Path_Cut = Path_Cuts / '2p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_2):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 3 (una permutación) {a, c, b, d}

    Path_Instance = Path_Instances / (ped.stem + '_3p.pkl')

    permutation_3 = permutation_vector(output_frames, 1, 2)

    instance = {'frames': permutation_3, 'class': 3}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación
    Path_Cut = Path_Cuts / '3p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_3):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 4 (una permutación) {c, b, a, d}

    Path_Instance = Path_Instances / (ped.stem + '_4p.pkl')

    permutation_4 = permutation_vector(output_frames, 0, 2)

    instance = {'frames': permutation_4, 'class': 4}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación
    Path_Cut = Path_Cuts / '4p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_4):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 5 (una permutación) {b, a, c, d}

    Path_Instance = Path_Instances / (ped.stem + '_5p.pkl')

    permutation_5 = permutation_vector(output_frames, 0, 1)

    instance = {'frames': permutation_5, 'class': 5}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación
    Path_Cut = Path_Cuts / '5p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_5):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 6 (una permutación) {a, c, d, b}

    Path_Instance = Path_Instances / (ped.stem + '_6p.pkl')

    permutation_6 = permutation_vector(permutation_1, 1, 3)

    instance = {'frames': permutation_6, 'class': 6}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación
    Path_Cut = Path_Cuts / '6p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_6):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 7 (una permutación) {a, d, b, c}

    Path_Instance = Path_Instances / (ped.stem + '_7p.pkl')

    permutation_7 = permutation_vector(permutation_2, 2, 3)

    instance = {'frames': permutation_7, 'class': 7}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación
    Path_Cut = Path_Cuts / '7p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_7):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 8 (una permutación) {c, a, b, d}

    Path_Instance = Path_Instances / (ped.stem + '_8p.pkl')

    permutation_8 = permutation_vector(permutation_3, 0, 1)

    instance = {'frames': permutation_8, 'class': 8}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación
    Path_Cut = Path_Cuts / '8p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_8):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 9 (una permutación) {b, c, a, d}

    Path_Instance = Path_Instances / (ped.stem + '_9p.pkl')

    permutation_9 = permutation_vector(permutation_4, 0, 1)

    instance = {'frames': permutation_9, 'class': 9}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación
    Path_Cut = Path_Cuts / '9p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_9):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 10 (una permutación) {b, a, d, c}

    Path_Instance = Path_Instances / (ped.stem + '_10p.pkl')

    permutation_10 = permutation_vector(permutation_5, 2, 3)

    instance = {'frames': permutation_10, 'class': 10}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación
    Path_Cut = Path_Cuts / '10p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_10):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)

    # Clase 11 (una permutación) {b, d, a, c}

    Path_Instance = Path_Instances / (ped.stem + '_11p.pkl')

    permutation_11 = permutation_vector(permutation_7, 0, 2)

    instance = {'frames': permutation_11, 'class': 11}

    with Path_Instance.open('wb') as file_descriptor:
        pickle.dump(instance, file_descriptor)

    # Se almacenan los recortes correspondientes a la permutación
    Path_Cut = Path_Cuts / '11p'

    Path_Cut.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(permutation_11):
        cv2.imwrite(str(Path_Cut / ('%01d' % (index) + '.jpg')), frame)


def permutation_vector(v, pos_1, pos_2):

    v_aux = v.copy()

    v_aux[pos_1], v_aux[pos_2] = v_aux[pos_2], v_aux[pos_1]

    return v_aux


########################################################################################################################
########################################################################################################################
########################################################################################################################


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

    PATH_cuts = Path(join(output_path_cuts, ID_set, ID_video, ID_pedestrian))
    PATH_cuts.mkdir(parents=True, exist_ok=True)

    for id_frame in range(total_frames):

        if id_frame == 0 or id_frame % frame_step == 0 or id_frame == (original_total - 1):

            frames.append(input_frames[id_frame])

            # Se almacenan los frames del video para visualizar cuales han sido seleccionados en cada caso
            cv2.imwrite(str(PATH_cuts / (str(id_frame) + '.jpg')), input_frames[id_frame])

        if len(frames) == n_frames_extracted:
            break

    return np.array(frames)


def save_frames(path_saves, frames, ID):

    if not os.path.exists(path_saves):
        os.mkdir(path_saves)

    for i in range(len(frames)):
        cv2.imwrite(path_saves + '/' + str(ID) + "__" + str(i) + ".jpg", frames[i])

