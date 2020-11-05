#LIMITAR CPU AL 45%
import os, sys
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45

#Se carga el fichero de configuración
import yaml

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
rootdir = os.path.dirname(parentparentdir)

with open(os.path.join(rootdir, 'config.yaml'), 'r') as file_descriptor:
    config = yaml.load(file_descriptor, Loader=yaml.FullLoader)

"""Inicialización de los generadores de números aleatorios. Se hace al inicio del codigo para evitar que el importar
otras librerias ya inicializen sus propios generadores"""

if not config['HP_Optimization_CrossingDetection_OrderPrediction']['random']:

    SEED = config['HP_Optimization_CrossingDetection_OrderPrediction']['seed']
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

sys.path.append(os.path.join(rootdir, 'utilities'))
sys.path.append(os.path.join(rootdir, 'Downstream_Tasks', 'CrossingDetection', 'OrderPrediction'))

import pickle
#import logging
from os.path import join
import time
from pathlib import Path
import json


import HyperModels_CrossingDetection_OrderPrediction, Tuners_CrossingDetection_OrderPrediction
import models_CrossingDetection_OrderPrediction
import DataGenerators_CrossingDetection_OrderPrediction

from FuncionesAuxiliares import read_instance_file_txt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from kerastuner.tuners import BayesianOptimization, Hyperband, RandomSearch


dim = config['HP_Optimization_CrossingDetection_OrderPrediction']['dim']
n_channels = config['HP_Optimization_CrossingDetection_OrderPrediction']['n_channels']

dataset = config['HP_Optimization_CrossingDetection_OrderPrediction']['dataset']
type_model = config['HP_Optimization_CrossingDetection_OrderPrediction']['type_model']
data_sampling = config['HP_Optimization_CrossingDetection_OrderPrediction']['data_sampling']

path_instances = Path(join(config['HP_Optimization_CrossingDetection_OrderPrediction']['path_instances'], dataset, 'CrossingDetection', '4_frames', data_sampling))
path_id_instances = Path(join(config['HP_Optimization_CrossingDetection_OrderPrediction']['path_id_instances'], dataset))

train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')
validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')

tuner_type = config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['type']
project_name = config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['project_name']
tuner_type_pretext_task = config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner_type_pretext_task']
project_name_pretext_task = config['HP_Optimization_CrossingDetection_OrderPrediction']['project_name_pretext_task']

epochs = config['HP_Optimization_CrossingDetection_OrderPrediction']['epochs']

# AÑADIR A ESTOS DIRECTORIOS EL MODELO FINAL PARA EL CUÁL SE ESTA CALCULANDO

path_output_results_CL = Path(join(config['HP_Optimization_CrossingDetection_OrderPrediction']['path_dir_results'], dataset, 'Transfer_Learning', 'CrossingDetection', 'OrderPrediction', tuner_type, type_model, 'Classification_Layer'))

path_output_results_FT = Path(join(config['HP_Optimization_CrossingDetection_OrderPrediction']['path_dir_results'], dataset, 'Transfer_Learning', 'CrossingDetection', 'OrderPrediction', tuner_type, type_model, 'Fine_Tuning'))

path_output_hyperparameters_CL = Path(join(config['HP_Optimization_CrossingDetection_OrderPrediction']['path_hyperparameters'], dataset, 'Transfer_Learning', 'CrossingDetection', 'OrderPrediction', tuner_type, type_model, 'Classification_Layer'))

path_output_hyperparameters_FT = Path(join(config['HP_Optimization_CrossingDetection_OrderPrediction']['path_hyperparameters'], dataset, 'Transfer_Learning', 'CrossingDetection', 'OrderPrediction', tuner_type, type_model, 'Fine_Tuning'))

path_output_results_CL.mkdir(parents=True, exist_ok=True)

path_output_results_FT.mkdir(parents=True, exist_ok=True)

path_output_hyperparameters_CL.mkdir(parents=True, exist_ok=True)

path_output_hyperparameters_FT.mkdir(parents=True, exist_ok=True)

#Ruta en la que se encuentran los pesos que se van a cargar en la capa convolucional del modelo
path_weights = Path(config['HP_Optimization_CrossingDetection_OrderPrediction']['path_weights'], dataset, 'OrderPrediction', data_sampling, tuner_type_pretext_task, type_model, project_name_pretext_task, 'weights.h5')


if type_model == 'SIAMESE':

    hypermodel_cl = HyperModels_CrossingDetection_OrderPrediction.HyperModel_OrderPrediction_SIAMESE_CrossingDetection_CL(the_input_shape=(dim[0], dim[1], n_channels), num_classes=2, path_weights=path_weights)


#SE DECLARA EL TUNER EN FUNCIÓN DE SU TIPO, DEL MODELO FINAL Y DE LA TAREA DE PRETEXTO
if tuner_type == 'Random_Search':

    tuner = Tuners_CrossingDetection_OrderPrediction.TunerRandomCrossingDetectionOrderPrediction(
        hypermodel_cl,
        objective=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['objetive'],
        seed=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['seed'],
        max_trials=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['max_trials'],
        executions_per_trial=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['executions_per_trial'],
        directory=path_output_results_CL,
        project_name=project_name,
        overwrite=False
    )

elif tuner_type == 'HyperBand':

    tuner = Tuners_CrossingDetection_OrderPrediction.TunerHyperBandCrossingDetectionOrderPrediction(
        hypermodel_cl,
        objective=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['objetive'],
        seed=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['seed'],
        max_epochs=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['max_epochs'],
        executions_per_trial=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['executions_per_trial'],
        directory=path_output_results_CL,
        project_name=project_name,
        overwrite=False
    )

else:

    tuner = Tuners_CrossingDetection_OrderPrediction.TunerBayesianCrossingDetectionOrderPrediction(
        hypermodel_cl,
        objective=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['objetive'],
        seed=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['seed'],
        max_trials=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['max_trials'],
        num_initial_points=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['num_initial_points'],
        directory=path_output_results_CL,
        project_name=project_name,
        overwrite=False
    )

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

tuner.search_space_summary()

start_time = time.time()

tuner.search(train_ids_instances, validation_ids_instances, dim, path_instances, 2, n_channels, 1, epochs, [earlystopping, reducelronplateau])

stop_time = time.time()

elapsed_time = stop_time - start_time

tuner.results_summary()

best_hp = tuner.get_best_hyperparameters()[0].values

# Se almacena el tuner en un fichero binario
with (path_output_results_CL / project_name / 'tuner.pkl').open('wb') as file_descriptor:
    pickle.dump(tuner, file_descriptor)

time_search_file = (path_output_results_CL / project_name / 'search_time.txt')

#Si el fichero con el tiempo de busqueda no existe, se crea
if not time_search_file.exists():

    with time_search_file.open('w') as filehandle:
        filehandle.write("Tiempo de busqueda: %f\n" % elapsed_time)

# Se guardan los hiperparámetros
with (path_output_hyperparameters_CL / (project_name + '.json')).open('w') as filehandle:
    json.dump(best_hp, filehandle)


#CON LOS HIPERPARÁMETROS OBTENIDOS SE ENTRENA EL MODELO QUE SE VA A USAR EN FINE TUNNING PARA OBTENER LOS PESOS DE LAS
#CAPAS DE CLASIFICACIÓN

if type_model == 'SIAMESE':

    best_model = models_CrossingDetection_OrderPrediction.model_CrossingDetection_OrderPrediction_SIAMESE(the_input_shape=(dim[0], dim[1], n_channels), units_dense_layer_1=best_hp['units_dense_layers_1'],
                                                                                                          units_dense_layer_2=best_hp['units_dense_layers_2'], learning_rate=best_hp['learning_rate'])

# Falta cargar los pesos obtenidos en el entrenamiento de la tarea de pretexto
best_model.load_weights(str(path_weights), by_name=True)

params = {
    'dim': dim,
    'path_instances': path_instances,
    'batch_size': best_hp['batch_size'],
    'n_clases': 2,
    'n_channels': n_channels,
    'normalized': best_hp['normalized'],
    'shuffle': best_hp['shuffle'],
}

train_generator = DataGenerators_CrossingDetection_OrderPrediction.DataGeneratorCrossingDetectionOrderPrediction(train_ids_instances, **params)
validation_generator = DataGenerators_CrossingDetection_OrderPrediction.DataGeneratorCrossingDetectionOrderPredictio(validation_ids_instances, **params)


earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

keras_callbacks = [earlystopping, reducelronplateau]

#ENTRENAMIENTO

best_model.fit(x=train_generator, validation_data=validation_generator, epochs=epochs, callbacks=keras_callbacks)

#Se guardan los pesos del modelo que va a ser utilizado en el ajuste fino
best_model.save_weights(str(path_output_results_CL / project_name / 'weights.h5'))

path_weights = path_output_results_CL / project_name / 'weights.h5'

# Fine Tuning

if type_model == 'SIAMESE':

    hypermodel_ft = HyperModels_CrossingDetection_OrderPrediction.HyperModel_OrderPrediction_SIAMESE_CrossingDetection_FT(the_input_shape=(dim[0], dim[1], n_channels), num_classes=2, path_weights=path_weights, hyperparameters=best_hp)

if tuner_type == 'Random_Search':

    tuner = RandomSearch(
        hypermodel_ft,
        objective=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['objetive'],
        seed=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['seed'],
        max_trials=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['max_trials'],
        executions_per_trial=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['executions_per_trial'],
        directory=path_output_results_FT,
        project_name=project_name,
        overwrite=False
    )

elif tuner_type == 'HyperBand':

    tuner = Hyperband(
        hypermodel_ft,
        objective=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['objetive'],
        seed=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['seed'],
        max_epochs=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['max_epochs'],
        executions_per_trial=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['executions_per_trial'],
        directory=path_output_results_FT,
        project_name=project_name,
        overwrite=False
    )

else:

    tuner = BayesianOptimization(
        hypermodel_ft,
        objective=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['objetive'],
        seed=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['seed'],
        max_trials=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['max_trials'],
        num_initial_points=config['HP_Optimization_CrossingDetection_OrderPrediction']['tuner']['num_initial_points'],
        directory=path_output_results_FT,
        project_name=project_name,
        overwrite=False
    )


earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

tuner.search_space_summary()

start_time = time.time()

tuner.search(x=train_generator, validation_data=validation_generator, epochs=epochs, callbacks=[earlystopping, reducelronplateau])

stop_time = time.time()

elapsed_time = stop_time - start_time

tuner.results_summary()

best_hp = tuner.get_best_hyperparameters()[0].values

# Se almacena el tuner en un fichero binario
with (path_output_results_FT / project_name / 'tuner.pkl').open('wb') as file_descriptor:
    pickle.dump(tuner, file_descriptor)

time_search_file = (path_output_results_FT / project_name / 'search_time.txt')

#Si el fichero con el tiempo de busqueda no existe, se crea
if not time_search_file.exists():

    with time_search_file.open('w') as filehandle:
        filehandle.write("Tiempo de busqueda: %f\n" % elapsed_time)


# Se guardan los hiperparámetros
with (path_output_hyperparameters_FT / (project_name + '.json')).open('w') as filehandle:
    json.dump(best_hp, filehandle)