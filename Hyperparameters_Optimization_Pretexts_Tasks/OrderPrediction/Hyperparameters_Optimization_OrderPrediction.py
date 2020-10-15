#LIMITAR CPU AL 45%
import os, sys
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45

#Se carga el fichero de configuración
import yaml

with open('../../config.yaml', 'r') as file_descriptor:
    config = yaml.load(file_descriptor, Loader=yaml.FullLoader)

"""Inicialización de los generadores de números aleatorios. Se hace al inicio del codigo para evitar que el importar
otras librerias ya inicializen sus propios generadores"""

if not config['Hyperparameters_Optimization_OrderPrediction']['random']:

    SEED = config['Hyperparameters_Optimization_OrderPrediction']['seed']
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

currentdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(rootdir, 'utilities'))

from os.path import join
import pickle
import time
from pathlib import Path
import json
#import logging

import HyperModels_Shuffle, Tuners_Shuffle

from FuncionesAuxiliares import read_instance_file_txt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


#logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

dataset = config['Hyperparameters_Optimization_OrderPrediction']['dataset']
type_model = config['Hyperparameters_Optimization_OrderPrediction']['type_model']
data_sampling = config['Hyperparameters_Optimization_OrderPrediction']['data_sampling']

path_instances = Path(join(config['Hyperparameters_Optimization_OrderPrediction']['path_instances'], dataset, 'OrderPrediction', data_sampling))
path_id_instances = Path(join(config['Hyperparameters_Optimization_OrderPrediction']['path_id_instances'], dataset))

#Se cargan los identificadores correspondientes a las instancias de entrenamiento y validación
train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')
validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')


tuner_type = config['Hyperparameters_Optimization_OrderPrediction']['tuner']['type']
project_name = config['Hyperparameters_Optimization_OrderPrediction']['tuner']['project_name']


dim = config['Hyperparameters_Optimization_OrderPrediction']['dim']
epochs = config['Hyperparameters_Optimization_OrderPrediction']['epochs']
n_frames = config['Hyperparameters_Optimization_OrderPrediction']['n_frames']
n_classes = config['Hyperparameters_Optimization_OrderPrediction']['n_classes']
n_channels = config['Hyperparameters_Optimization_OrderPrediction']['n_channels']


path_output_results = Path(join(config['Hyperparameters_Optimization_OrderPrediction']['path_dir_results'], dataset, 'OrderPrediction', data_sampling, tuner_type, type_model))

path_output_hyperparameters = Path(join(config['Hyperparameters_Optimization_OrderPrediction']['path_hyperparameters'], dataset, 'OrderPrediction', data_sampling, tuner_type, type_model))

#Se crean los directorios donde almacenar los resultados
path_output_results.mkdir(parents=True, exist_ok=True)
#Se crean los directorios para almacenar los hiperparametros
path_output_hyperparameters.mkdir(parents=True, exist_ok=True)

#SE DEFINE EL HYPERMODELO EN FUNCIÓN DEL TIPO DE MODELO
if type_model == 'SIAMESE':

    hypermodel = HyperModels_Shuffle.HyperModel_OrderPrediction_SIAMESE(the_input_shape=(dim[0], dim[1], n_channels), num_classes=n_classes)

#SE DEFINE EL TUNER EN FUNCIÓN DE SU TIPO
if tuner_type == 'Random_Search':

    tuner = Tuners_Shuffle.TunerRandomShuffle(
        hypermodel,
        objective=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['objetive'],
        seed=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['seed'],
        max_trials=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['max_trials'],
        executions_per_trial=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['executions_per_trial'],
        directory=path_output_results,
        project_name=project_name,
        overwrite=False
    )

elif tuner_type == 'HyperBand':

    tuner = Tuners_Shuffle.TunerHyperBandShuffle(
        hypermodel,
        objective=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['objetive'],
        seed=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['seed'],
        max_epochs=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['max_epochs'],
        executions_per_trial=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['executions_per_trial'],
        directory=path_output_results,
        project_name=project_name,
        overwrite=False
    )

else:

    tuner = Tuners_Shuffle.TunerBayesianShuffle(
        hypermodel,
        objective=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['objetive'],
        seed=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['seed'],
        max_trials=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['max_trials'],
        num_initial_points=config['Hyperparameters_Optimization_OrderPrediction']['tuner']['num_initial_points'],
        directory=path_output_results,
        project_name=project_name,
        overwrite=False
    )

#SE LLEVA A CABO LA BUSQUEDA DE LOS MEJORES HIPERPARÁMETROS
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

tuner.search_space_summary()

start_time = time.time()

tuner.search(train_ids_instances, validation_ids_instances, dim, path_instances, n_classes, n_channels, 1, epochs, [earlystopping, reducelronplateau])

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