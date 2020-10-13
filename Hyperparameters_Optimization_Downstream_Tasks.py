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

if not config['Hyperparameters_Optimization_Downstream_Tasks']['random']:

    SEED = config['Hyperparameters_Optimization_Downstream_Tasks']['seed']
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

import HyperModels_Transfer_Learning
from FuncionesAuxiliares import read_instance_file_txt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

import Tuners_Transfer_Learning

from kerastuner.tuners import BayesianOptimization, Hyperband, RandomSearch

import pickle

#import logging

from os.path import join

import time

import DataGenerators_Pretext_Tasks

from pathlib import Path

import json


dataset = config['Hyperparameters_Optimization_Downstream_Tasks']['dataset']
pretext_task = config['Hyperparameters_Optimization_Downstream_Tasks']['pretext_task']
type_model = config['Hyperparameters_Optimization_Downstream_Tasks']['type_model']
data_sampling = config['Hyperparameters_Optimization_Downstream_Tasks']['data_sampling']
downstream_task = config['Hyperparameters_Optimization_Downstream_Tasks']['downstream_task']

n_frames = config['Hyperparameters_Optimization_Downstream_Tasks']['n_frames']

path_instances = Path(join(config['Hyperparameters_Optimization_Downstream_Tasks'], dataset, downstream_task, str(n_frames) + '_frames', data_sampling))

path_id_instances = Path(join(config['Hyperparameters_Optimization_Downstream_Tasks']['path_id_instances'], dataset))

train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')
validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')

tuner_type = config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['type']
project_name = config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['project_name']
tuner_type_pretext_task = config['Hyperparameters_Optimization_Downstream_Tasks']['tuner_type_pretext_task']
project_name_pretext_task = config['Hyperparameters_Optimization_Downstream_Tasks']['project_name_pretext_task']

dim = config['Hyperparameters_Optimization_Downstream_Tasks']['dim']
epochs = config['Hyperparameters_Optimization_Downstream_Tasks']['epochs']
n_classes = config['Hyperparameters_Optimization_Downstream_Tasks']['n_classes']
n_channels = config['Hyperparameters_Optimization_Downstream_Tasks']['n_channels']


# AÑADIR A ESTOS DIRECTORIOS EL MODELO FINAL PARA EL CUÁL SE ESTA CALCULANDO

path_output_results_CL = Path(join(config['Hyperparameters_Optimization_Downstream_Tasks']['path_dir_results'], dataset, 'Transfer_Learning', downstream_task, pretext_task, tuner_type, type_model, 'Classification_Layer'))

path_output_results_FT = Path(join(config['Hyperparameters_Optimization_Downstream_Tasks']['path_dir_results'], dataset, 'Transfer_Learning', downstream_task, pretext_task, tuner_type, type_model, 'Fine_Tuning'))

path_output_hyperparameters_CL = Path(join(config['Hyperparameters_Optimization_Downstream_Tasks']['path_hyperparameters'], dataset, 'Transfer_Learning', downstream_task, pretext_task, tuner_type, type_model, 'Classification_Layer'))

path_output_hyperparameters_FT = Path(join(config['Hyperparameters_Optimization_Downstream_Tasks']['path_hyperparameters'], dataset, 'Transfer_Learning', downstream_task, pretext_task, tuner_type, type_model, 'Fine_Tuning'))

path_output_results_CL.mkdir(parents=True, exist_ok=True)

path_output_results_FT.mkdir(parents=True, exist_ok=True)

path_output_hyperparameters_CL.mkdir(parents=True, exist_ok=True)

path_output_hyperparameters_FT.mkdir(parents=True, exist_ok=True)

#Ruta en la que se encuentran los pesos que se van a cargar en la capa convolucional del modelo
path_weights = Path(config['Hyperparameters_Optimization_Downstream_Tasks']['path_weights'], dataset, pretext_task, data_sampling, tuner_type_pretext_task, type_model, project_name_pretext_task)

#DEFINICIÓN DEL HIPERMODELO EN FUNCIÓN DE LA TAREA DE PRETEXTO, EL TIPO DE MODELO Y DEL MODELO FINAL
if pretext_task == 'Shuffle':

    if type_model == 'CONV3D':

        if downstream_task == 'Crossing-detection':

            hypermodel_cl = HyperModels_Transfer_Learning.HyperModel_FINAL_Shuffle_CONV3D_CrossingDetection_CL(the_input_shape=(n_frames, dim[0], dim[1], 3), num_classes=n_classes, path_weights=path_weights)

elif pretext_task == 'OrderPrediction':

    if type_model == 'SIAMESE':

        if downstream_task == 'Crossing-detection':

            hypermodel_cl = HyperModels_Transfer_Learning.HyperModel_FINAL_OrderPrediction_SIAMESE_CrossingDetection_CL(the_input_shape=(128, 128, 3), num_classes=n_classes, path_weights=path_weights)


#SE DECLARA EL TUNER EN FUNCIÓN DE SU TIPO, DEL MODELO FINAL Y DE LA TAREA DE PRETEXTO
if tuner_type == 'Random_Search':

    if downstream_task == 'Crossing-detection':

        if pretext_task == 'Shuffle':

            tuner = Tuners_Transfer_Learning.TunerRandomFINALCrossingDetectionShuffle(
                hypermodel_cl,
                objective=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['objetive'],
                seed=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['seed'],
                max_trials=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['max_trials'],
                executions_per_trial=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['executions_per_trial'],
                directory=path_output_results_CL,
                project_name=project_name,
                overwrite=False
            )

        elif pretext_task == 'OrderPrediction':

            tuner = Tuners_Transfer_Learning.TunerRandomFINALCrossingDetectionOrderPrediction(
                hypermodel_cl,
                objective=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['objetive'],
                seed=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['seed'],
                max_trials=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['max_trials'],
                executions_per_trial=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['executions_per_trial'],
                directory=path_output_results_CL,
                project_name=project_name,
                overwrite=False
            )

elif tuner_type == 'HyperBand':

    if downstream_task == 'Crossing-detection':

        if pretext_task == 'Shuffle':

            tuner = Tuners_Transfer_Learning.TunerHyperBandFINALCrossingDetectionShuffle(
                hypermodel_cl,
                objective=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['objetive'],
                seed=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['seed'],
                max_epochs=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['max_epochs'],
                executions_per_trial=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['executions_per_trial'],
                directory=path_output_results_CL,
                project_name=project_name,
                overwrite=False
            )

        elif pretext_task == 'OrderPrediction':

            tuner = Tuners_Transfer_Learning.TunerHyperBandFINALCrossingDetectionOrderPrediction(
                hypermodel_cl,
                objective=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['objetive'],
                seed=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['seed'],
                max_epochs=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['max_epochs'],
                executions_per_trial=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['executions_per_trial'],
                directory=path_output_results_CL,
                project_name=project_name,
                overwrite=False
            )

else:

    if downstream_task == 'Crossing-detection':

        if pretext_task == 'Shuffle':

            tuner = Tuners_Transfer_Learning.TunerBayesianFINALCrossingDetectionShuffle(
                hypermodel_cl,
                objective=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['objetive'],
                seed=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['seed'],
                max_trials=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['max_trials'],
                num_initial_points=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['num_initial_points'],
                directory=path_output_results_CL,
                project_name=project_name,
                overwrite=False
            )

        elif pretext_task == 'OrderPrediction':

            tuner = Tuners_Transfer_Learning.TunerBayesianFINALCrossingDetectionOrderPrediction(
                hypermodel_cl,
                objective=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['objetive'],
                seed=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['seed'],
                max_trials=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['max_trials'],
                num_initial_points=config['Hyperparameters_Optimization_Downstream_Tasks']['tuner']['num_initial_points'],
                directory=path_output_results_CL,
                project_name=project_name,
                overwrite=False
            )

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min',
                                      min_delta=0.0001, cooldown=0, min_lr=0)

# !!!!!!PONER reducelronplateau COMO CALLBACKS Y AJUSTAR SUS HIPERPARÁMETROS!!!!!!!!!!!!!!

tuner.search_space_summary()

start_time = time.time()

if pretext_task == 'Shuffle':

    tuner.search(train_ids_instances, validation_ids_instances, dim, path_instances, n_frames, n_classes, n_channels, 1, epochs, [earlystopping, reducelronplateau])

elif pretext_task == 'OrderPrediction':

    tuner.search(train_ids_instances, validation_ids_instances, dim, path_instances, n_classes, n_channels, 1, epochs, [earlystopping, reducelronplateau])

stop_time = time.time()

elapsed_time = stop_time - start_time

tuner.results_summary()

best_hp = tuner.get_best_hyperparameters()[0].values

# Se almacena el tuner en un fichero binario
with (path_output_results_CL / project_name / 'tuner.pkl').open('wb') as file_descriptor:
    pickle.dump(tuner, file_descriptor)

with (path_output_results_CL / project_name / 'search_time.txt').open('w') as filehandle:
    filehandle.write("Tiempo de busqueda: %f\n" % elapsed_time)

# Se guardan los hiperparámetros
with (path_output_hyperparameters_CL / (project_name + '.json')).open('w') as filehandle:
    json.dump(best_hp, filehandle)

# Fine Tuning

if pretext_task == 'Shuffle':

    if type_model == 'CONV3D':

        if downstream_task == 'Crossing-detection':

            hypermodel_ft = HyperModels_Transfer_Learning.HyperModel_FINAL_Shuffle_CONV3D_CrossingDetection_FT(input_shape=(n_frames, dim[0], dim[1], 3), num_classes=n_classes, path_weights=path_weights, hyperparameters=best_hp)

if tuner_type == 'Random_Search':

    tuner = RandomSearch(
        hypermodel_ft,
        objective=config['Keras_Tuner']['tuner']['objetive'],
        seed=config['Keras_Tuner']['tuner']['seed'],
        max_trials=config['Keras_Tuner']['tuner']['max_trials'],
        executions_per_trial=config['Keras_Tuner']['tuner']['executions_per_trial'],
        directory=path_output_results_FT,
        project_name=project_name,
        overwrite=False
    )

elif tuner_type == 'HyperBand':

    tuner = Hyperband(
        hypermodel_ft,
        objective=config['Keras_Tuner']['tuner']['objetive'],
        seed=config['Keras_Tuner']['tuner']['seed'],
        max_epochs=config['Keras_Tuner']['tuner']['max_epochs'],
        executions_per_trial=config['Keras_Tuner']['tuner']['executions_per_trial'],
        directory=path_output_results_FT,
        project_name=project_name,
        overwrite=False
    )

else:

    tuner = BayesianOptimization(
        hypermodel_ft,
        objective=config['Keras_Tuner']['tuner']['objetive'],
        seed=config['Keras_Tuner']['tuner']['seed'],
        max_trials=config['Keras_Tuner']['tuner']['max_trials'],
        num_initial_points=config['Keras_Tuner']['tuner']['num_initial_points'],
        directory=path_output_results_FT,
        project_name=project_name,
        overwrite=False
    )

params = {
    'dim': dim,
    'path_instances': path_instances,
    'batch_size': best_hp['batch_size'],
    'n_clases': n_classes,
    'n_channels': n_channels,
    'n_frames': n_frames,
    'normalized': best_hp['normalized'],
    'shuffle': best_hp['normalized'],
}

if type_model == 'Crossing-detection':
    train_generator = DataGenerators_Pretext_Tasks.DataGeneratorFINALCrossingDetection(train_ids_instances, **params)
    validation_generator = DataGenerators_Pretext_Tasks.DataGeneratorFINALCrossingDetection(validation_ids_instances, **params)

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min',
                              restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min',
                                      min_delta=0.0001, cooldown=0, min_lr=0)

tuner.search_space_summary()

start_time = time.time()

tuner.search(x=train_generator, validation_data=validation_generator, epochs=epochs,
             callbacks=[earlystopping, reducelronplateau])

stop_time = time.time()

elapsed_time = stop_time - start_time

tuner.results_summary()

best_hp = tuner.get_best_hyperparameters()[0].values

# Se almacena el tuner en un fichero binario
with (path_output_results_FT / project_name / 'tuner.pkl').open('wb') as file_descriptor:
    pickle.dump(tuner, file_descriptor)

with (path_output_results_FT / project_name / 'search_time.txt').open('w') as filehandle:
    filehandle.write("Tiempo de busqueda: %f\n" % elapsed_time)

# Se guardan los hiperparámetros
with (path_output_hyperparameters_FT / (project_name + '.json')).open('w') as filehandle:
    json.dump(best_hp, filehandle)