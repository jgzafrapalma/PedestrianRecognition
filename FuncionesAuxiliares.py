import math
import cv2
import numpy as np
import os
from os.path import isfile, join
import random
import pickle
import errno
import copy

import argparse

from tensorflow.keras.preprocessing.image import img_to_array


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

def create_Train_Validation(path_instances, percent_Validation):
    #Me quedo unicamente con el nombre de las instancias sin el formato
    onlyfiles = [f for f in os.listdir(path_instances) if isfile(join(path_instances, f))]

    train, validation = np.split(onlyfiles, [int(len(onlyfiles)*(1 - percent_Validation))])

    return train.tolist(), validation.tolist()

#pathVideos: ruta donde se encuentran los videos
#pathInsatnces: ruta donde se quiere almacenar las instancias que se van a generar
#pathFrames: ruta donde se van a almacenar los frames del problema si la variable booleana frames esta activa

def extract_pedestrians_datasets(pathVideos, pathInstances, pathFrames, pathData, rate, shape=(), frames=True):

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
            cap = cv2.VideoCapture(pathVideos + f)

            print(pathVideos + f)

            #Nombre del fichero sin extención
            f_no_ext = ".".join(f.split(".")[:-1])

            if frames:
                if not os.path.exists(pathFrames + f_no_ext):
                    os.mkdir(pathFrames + f_no_ext)

            width = data[f_no_ext]['width']
            height = data[f_no_ext]['height']

            #Lista con los peatones (personas que interactuan con el conductor) del video f_no_ext
            list_pedestrian = [ped for ped in list(data[f_no_ext]['ped_annotations']) if data[f_no_ext]['ped_annotations'][ped]['old_id'].find('pedestrian') != -1]

            #Lista donde se van a ir almacenando las matrices correspondientes a cada unos de los peatones del video f
            cuts_pedestrian = list()

            # Se reserva espacio para almacenar los distintos frames recortados de cada uno de los peatones
            for id_ped in list_pedestrian:
                #Numero de frames en los que aparece el peaton id_ped
                num_frames = len(data[f_no_ext]['ped_annotations'][id_ped]['frames'])
                cuts_pedestrian.append(np.zeros((num_frames, shape[0], shape[1], 3)))

            id_frame = 0
            while cap.isOpened():

                ret, frame = cap.read()

                #Si no se puede abrir el video se sale del bucle while
                if not ret:
                    break

                for id_ped, ped in enumerate(list_pedestrian):

                    if not os.path.exists(pathFrames + f_no_ext + '/' + ped):
                        os.mkdir(pathFrames + f_no_ext + '/' + ped)

                    #Compruebo si el peaton se encuentra en el frame actual
                    list_frames = data[f_no_ext]['ped_annotations'][ped]['frames']
                    if id_frame in list_frames:
                        #Obtengo la posición de la lista de frames del peaton en el que es encuentra en frame actual.
                        #(Va a servir para luego saber en que index consultar la bounding box)
                        index_frame = list_frames.index(id_frame)
                        #Lista con las coordendas de los dos puntos de la bounding box
                        bbox = data[f_no_ext]['ped_annotations'][ped]['bbox'][index_frame]

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

                        if frame:
                            cv2.imwrite(pathFrames + f_no_ext + '/' + ped + '/' + '%03d' % id_frame + '.jpg', cut)

                        cuts_pedestrian[id_ped][index_frame] = cut

                id_frame += 1

            cap.release()

            for id_ped, cut_predestrian in enumerate(cuts_pedestrian):
                np.save(pathInstances + f_no_ext + '_' + list_pedestrian[id_ped] + '.npy', cuts_pedestrian[id_ped])



def save_frames(path_saves, frames, ID):

    if not os.path.exists(path_saves):
        os.mkdir(path_saves)

    for i in range(len(frames)):
        cv2.imwrite(path_saves + '/' + str(ID) + "__" + str(i) + ".jpg", frames[i])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pathIn', help="path to video")

    args = parser.parse_args()

    videos = os.listdir("./datasets/JAAD_clips/")

    #NINSTANCES = len(videos) * 2
    #NFRAMES = 5
    #WIDTH = 1920
    #HEIGHT = 1080
    #CHANNELS = 3

    # Por cada secuencia de frames (clase positiva), se va a generar otra desordenada (clase negativa)
    #Dataset = np.zeros((NINSTANCES, NFRAMES, HEIGHT, WIDTH, CHANNELS))

    #labels = np.zeros(len(videos) * 2)

    #for i in range(len(videos)):
    frames = extractFrames(args.pathIn, 8)

    frames_shuffle = ShuffleFrames(frames, 1)

    print(frames_shuffle.shape)

    for id_frame in range(8):
        filename = "./datasets/JAAD_clips/shuffle/" + str(id_frame) + ".jpg"

        cv2.imwrite(filename, frames_shuffle[id_frame])


    # path in -> "./datasets/JAAD_clips/video_0001.mp4"
