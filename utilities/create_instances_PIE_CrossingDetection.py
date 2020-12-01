#LIMITAR CPU AL 45%
import os, sys
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45

#Se carga el fichero de configuración
import yaml

currentdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.dirname(currentdir)

with open(os.path.join(rootdir, 'config.yaml'), 'r') as file_descriptor:
    config = yaml.load(file_descriptor, Loader=yaml.FullLoader)


from FuncionesAuxiliares import extract_pedestriansFrames_datasets_PIE, extractFramesOpticalFlow, extractFramesUniform

import logging
import pickle
from pathlib import Path
from os.path import join
import cv2



#Función que permite crear instancias que van a ser utilizadas para la evaluación del modelo final
def create_instances_PIE_CrossingDetection(input_path_data, input_path_dataset, output_path_frames, output_path_instances, output_path_cuts, n_frames, rate, shape, optical_flow=False):

    #Se comprueba si los frames necesarios para crear las instancias ya estan creados
    if not Path(join(output_path_frames, str(shape[0]) + '_' + str(shape[1]))).exists():

        extract_pedestriansFrames_datasets_PIE(input_path_data, input_path_dataset, output_path_frames, rate, shape)


    logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

    # Se abre el fichero de datos
    with open(input_path_data, 'rb') as file_descriptor:
        data = pickle.load(file_descriptor)

    # Ruta donde se encuentran los frames de cada peaton por video
    Path_Frames = Path(join(output_path_frames, str(shape[0]) + '_' + str(shape[1])))

    for set_video in Path_Frames.iterdir():

        logging.info("Accediendo al directorio %s" % set_video)

        for video in set_video.iterdir():

            logging.info("Creando instancias del video %s" % video)

            for ped in video.iterdir():

                Path_Ped = Path(join(str(ped), '%04d.jpg'))

                #Se obtiene la etiqueta de si el peatón esta cruzando o no
                crossing = data[set_video.stem][video.stem]['ped_annotations'][ped.stem]['attributes']['crossing']

                #Solamente se va a calcular para aquellos peatones que estan cruzando o no
                if crossing != -1:

                    if optical_flow:

                        Path_Instances = Path(join(output_path_instances, 'CrossingDetection', str(n_frames) + '_frames', str(shape[0]) + '_' + str(shape[1]), 'OpticalFlow'))

                        Path_Cuts = Path(join(output_path_cuts, 'CrossingDetection', str(n_frames) + '_frames', str(shape[0]) + '_' + str(shape[1]), 'OpticalFlow', ped.stem))

                        Path_Instances.mkdir(parents=True, exist_ok=True)

                        Path_Cuts.mkdir(parents=True, exist_ok=True)

                        # Comprobar si ya existe la instancia
                        Path_Instance = Path_Instances / (ped.stem + '.pkl')

                        if not Path_Instance.exists():

                            logging.info("Creando instancia %s" % Path_Instance.stem)

                            output_frames = extractFramesOpticalFlow(Path_Ped, n_frames)

                            # Se crea la instancia
                            instance = {'frames': output_frames, 'crossing': crossing}

                            with Path_Instance.open('wb') as file_descriptor:
                                pickle.dump(instance, file_descriptor)

                            for index, frame in enumerate(output_frames):
                                cv2.imwrite(str(Path_Cuts / ('%01d' % (index) + '.jpg')), frame)

                            logging.info("Instancia %s creada con exito" % Path_Instance.stem)

                        else:

                            logging.info("La instancia %s no se ha creado porque ya existe" % Path_Instance.stem)

                    else:

                        Path_Instances = Path(join(output_path_instances, 'CrossingDetection', str(n_frames) + '_frames', str(shape[0]) + '_' + str(shape[1]), 'Distributed'))

                        Path_Cuts = Path(join(output_path_cuts, 'CrossingDetection', str(n_frames) + '_frames', str(shape[0]) + '_' + str(shape[1]), 'Distributed', ped.stem))

                        Path_Instances.mkdir(parents=True, exist_ok=True)

                        Path_Cuts.mkdir(parents=True, exist_ok=True)

                        # Comprobar si ya existe la instancia
                        Path_Instance = Path_Instances / (ped.stem + '.pkl')

                        if not Path_Instance.exists():

                            output_frames = extractFramesUniform(Path_Ped, n_frames)

                            # Se crea la instancia
                            instance = {'frames': output_frames, 'crossing': crossing}

                            with Path_Instance.open('wb') as file_descriptor:
                                pickle.dump(instance, file_descriptor)

                            for index, frame in enumerate(output_frames):
                                cv2.imwrite(str(Path_Cuts / ('%01d' % (index) + '.jpg')), frame)

                            logging.info("Instancia %s creada con exito" % Path_Instance.stem)

                        else:

                            logging.info("La instancia %s no se ha creado porque ya existe" % Path_Instance.stem)

create_instances_PIE_CrossingDetection(
    input_path_data=config['create_instances_PIE_CrossingDetection']['input_path_data'],
    input_path_dataset=config['create_instances_PIE_CrossingDetection']['input_path_dataset'],
    output_path_frames=config['create_instances_PIE_CrossingDetection']['output_path_frames'],
    output_path_instances=config['create_instances_PIE_CrossingDetection']['output_path_instances'],
    output_path_cuts=config['create_instances_PIE_CrossingDetection']['output_path_cuts'],
    n_frames=config['create_instances_PIE_CrossingDetection']['n_frames'],
    rate=config['create_instances_PIE_CrossingDetection']['rate'],
    shape=config['create_instances_PIE_CrossingDetection']['shape'],
    optical_flow=config['create_instances_PIE_CrossingDetection']['optical_flow']
)
