#LIMITAR CPU AL 45%
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45

#Se carga el fichero de configuración
import yaml

with open('config.yaml', 'r') as file_descriptor:
    config = yaml.load(file_descriptor, Loader=yaml.FullLoader)

"""Inicialización de los generadores de números aleatorios. Se hace al inicio del codigo para evitar que el importar
otras librerias ya inicializen sus propios generadores"""

if not config['Keras_Tuner']['random']:

    SEED = config['Keras_Tuner']['seed']
    from numpy.random import seed
    seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from random import seed
    seed(SEED)

#############################################SOLUCIONAR EL ERROR DE LA LIBRERIA CUDNN###################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

configProto = ConfigProto()
configProto.gpu_options.allow_growth = True
session = InteractiveSession(config=configProto)

########################################################################################################################

import HyperModels_Pretext_Tasks
from FuncionesAuxiliares import read_instance_file_txt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

import Tuners_Pretext_Tasks

import pickle

#import logging

from os.path import join

import time

from pathlib import Path

import json


#logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

dataset = config['Hyperparameters_Optimization_Pretext_Tasks']['dataset']
pretext_task = config['Hyperparameters_Optimization_Pretext_Tasks']['pretext_task']
type_model = config['Hyperparameters_Optimization_Pretext_Tasks']['type_model']
data_sampling = config['Hyperparameters_Optimization_Pretext_Tasks']['type_model']

path_instances = Path(join(config['Hyperparameters_Optimization_Pretext_Tasks']['path_instances'], dataset, pretext_task, data_sampling))
path_id_instances = Path(join(config['Hyperparameters_Optimization_Pretext_Tasks']['path_id_instances'], dataset))

#Se cargan los identificadores correspondientes a las instancias de entrenamiento y validación
train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')
validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')


tuner_type = config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['type']
project_name = config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['project_name']


dim = config['Hyperparameters_Optimization_Pretext_Tasks']['dim']
epochs = config['Hyperparameters_Optimization_Pretext_Tasks']['epochs']
n_frames = config['Hyperparameters_Optimization_Pretext_Tasks']['n_frames']
num_classes = config['Hyperparameters_Optimization_Pretext_Tasks']['num_classes']
n_channels = config['Hyperparameters_Optimization_Pretext_Tasks']['n_channels']


path_output_results = Path(join(config['Keras_Tuner']['path_dir_results'], dataset, pretext_task, data_sampling,tuner_type, type_model))

path_output_hyperparameters = Path(join(config['Keras_Tuner']['path_hyperparameters'], dataset, pretext_task, data_sampling, tuner_type, type_model))

#Se crean los directorios donde almacenar los resultados
path_output_results.mkdir(parents=True, exist_ok=True)
#Se crean los directorios para almacenar los hiperparametros
path_output_hyperparameters.mkdir(parents=True, exist_ok=True)

#SE DEFINE EL TUNER
if pretext_task == 'Shuffle':
    if type_model == 'CONV3D':
        hypermodel = HyperModels_Pretext_Tasks.HyperModelShuffleCONV3D(input_shape=(n_frames, dim[0], dim[1], 3), num_classes=num_classes)
elif pretext_task == 'OrderPrediction':
    if type_model == 'SIAMESE':
        hypermodel = HyperModels_Pretext_Tasks.HyperModel_OrderPrediction_SIAMESE(input_shape=(n_frames, dim[0], dim[1], 3), num_classes=num_classes)



if tuner_type == 'Random_Search':

    if pretext_task == 'Shuffle':

        tuner = Tuners_Pretext_Tasks.TunerRandomShuffle(
            hypermodel,
            objective=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['objetive'],
            seed=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['seed'],
            max_trials=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['max_trials'],
            executions_per_trial=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['executions_per_trial'],
            directory=path_output_results,
            project_name=project_name,
            overwrite=False
        )

elif tuner_type == 'HyperBand':

    if pretext_task == 'Shuffle':

        tuner = Tuners_Pretext_Tasks.TunerHyperBandShuffle(
            hypermodel,
            objective=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['objetive'],
            seed=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['seed'],
            max_epochs=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['max_epochs'],
            executions_per_trial=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['executions_per_trial'],
            directory=path_output_results,
            project_name=project_name,
            overwrite=False
        )

else:

    if pretext_task == 'Shuffle':

        tuner = Tuners_Pretext_Tasks.TunerBayesianShuffle(
            hypermodel,
            objective=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['objetive'],
            seed=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['seed'],
            max_trials=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['max_trials'],
            num_initial_points=config['Hyperparameters_Optimization_Pretext_Tasks']['tuner']['num_initial_points'],
            directory=path_output_results,
            project_name=project_name,
            overwrite=False
        )

#SE LLEVA A CABO LA BUSQUEDA DE LOS MEJORES HIPERPARÁMETROS
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

tuner.search_space_summary()

start_time = time.time()

tuner.search(train_ids_instances, validation_ids_instances, dim, path_instances, n_frames, num_classes, n_channels, 1, epochs, [earlystopping, reducelronplateau])

stop_time = time.time()

elapsed_time = stop_time - start_time

tuner.results_summary()

best_hp = tuner.get_best_hyperparameters()[0].values

#Se almacena el tuner en un fichero binario
with (path_output_results / project_name / 'tuner.pkl').open('wb') as file_descriptor:
    pickle.dump(tuner, file_descriptor)

with (path_output_results / project_name / 'search_time.txt').open('w') as file_descriptor:
    file_descriptor.write("Tiempo de busqueda: %f\n" % elapsed_time)

#Se guardan los hiperparámetros
with (path_output_hyperparameters / (project_name + '.json')).open('w') as file_descriptor:
    json.dump(best_hp, file_descriptor)

