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


from FuncionesAuxiliares import extract_pedestriansFrames_datasets_PIE, extractFramesOpticalFlow, extractFramesUniform, create_permutations_OrderPrediction, permutation_vector

import logging
from pathlib import Path
from os.path import join




#Función que permite crear instancias que van a ser utilizadas para la evaluación del modelo final
def create_instances_PIE_OrderPrediction(input_path_data, input_path_dataset, output_path_frames, output_path_instances, output_path_cuts, rate, shape, optical_flow=False):

    #Se comprueba si los frames necesarios para crear las instancias ya estan creados
    if not Path(join(output_path_frames, str(shape[0]) + '_' + str(shape[1]))).exists():

        extract_pedestriansFrames_datasets_PIE(input_path_data, input_path_dataset, output_path_frames, rate, shape)


    logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

    # Ruta donde se encuentran los frames de cada peaton por video
    Path_Frames = Path(join(output_path_frames, str(shape[0]) + '_' + str(shape[1])))

    for set_video in Path_Frames.iterdir():

        logging.info("Accediendo al directorio %s" % set_video)

        for video in set_video.iterdir():

            logging.info("Creando instancias del video %s" % video)

            for ped in video.iterdir():

                Path_Ped = Path(join(str(ped), '%04d.jpg'))

                if optical_flow:

                    Path_Instances = Path(join(output_path_instances, 'OrderPrediction', str(shape[0]) + '_' + str(shape[1]), 'OpticalFlow'))

                    Path_Cuts = Path(join(output_path_cuts, 'OrderPrediction', 'OpticalFlow', str(shape[0]) + '_' + str(shape[1]), ped.stem))

                    output_frames = extractFramesOpticalFlow(Path_Ped, 4)

                else:

                    Path_Instances = Path(join(output_path_instances, 'OrderPrediction', str(shape[0]) + '_' + str(shape[1]), 'Distributed'))

                    Path_Cuts = Path(join(output_path_cuts, 'OrderPrediction', 'Distributed', str(shape[0]) + '_' + str(shape[1]), ped.stem))

                    output_frames = extractFramesUniform(Path_Ped, 4)

                Path_Instances.mkdir(parents=True, exist_ok=True)

                Path_Cuts.mkdir(parents=True, exist_ok=True)

                create_permutations_OrderPrediction(Path_Instances, Path_Cuts, ped, output_frames)

                logging.info("Permutaciones(instancias) para el peatón %s creadas con exitos" % ped.stem)

create_instances_PIE_OrderPrediction(
    input_path_data=config['create_instances_PIE_CrossingDetection']['input_path_data'],
    input_path_dataset=config['create_instances_PIE_CrossingDetection']['input_path_dataset'],
    output_path_frames=config['create_instances_PIE_CrossingDetection']['output_path_frames'],
    output_path_instances=config['create_instances_PIE_CrossingDetection']['output_path_instances'],
    output_path_cuts=config['create_instances_PIE_CrossingDetection']['output_path_cuts'],
    rate=config['create_instances_PIE_CrossingDetection']['rate'],
    shape=config['create_instances_PIE_CrossingDetection']['shape'],
    optical_flow=config['create_instances_PIE_CrossingDetection']['optical_flow']
)
