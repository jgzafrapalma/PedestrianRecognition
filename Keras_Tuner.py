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

import HyperModels
from FuncionesAuxiliares import read_instance_file_txt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

import Tuners

from kerastuner.tuners import BayesianOptimization, Hyperband, RandomSearch

import pickle

#import logging

from os.path import join

import time

import DataGenerators

from pathlib import Path

import json

def run_hyperparameter_tuning():

    #logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

    dataset = config['Keras_Tuner']['dataset']

    pretext_task = config['Keras_Tuner']['pretext_task']
    model_name = config['Keras_Tuner']['model']
    tuner_type = config['Keras_Tuner']['tuner']['type']
    project_name = config['Keras_Tuner']['tuner']['project_name']
    classification = config['Keras_Tuner']['classification']

    path_instances = Path(join(config['Keras_Tuner']['path_instances'], dataset))
    path_id_instances = Path(join(config['Keras_Tuner']['path_id_instances'], dataset))

    #Se cargan los identificadores correspondientes a las instancias de entrenamiento y validación
    train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')
    validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')


    dim = config['Keras_Tuner']['dim']
    epochs = config['Keras_Tuner']['epochs']
    n_frames = config['Keras_Tuner']['n_frames']
    num_classes = config['Keras_Tuner']['num_classes']

    Transfer_Learning = config['Keras_Tuner']['Transfer_Learning']

    if not Transfer_Learning:

        path_output_results = Path(join(config['Keras_Tuner']['path_dir_results'], dataset, pretext_task, tuner_type, model_name))

        path_output_hyperparameters = Path(join(config['Keras_Tuner']['path_hyperparameters'], dataset, pretext_task, tuner_type, model_name))

        #Se crean los directorios donde almacenar los resultados
        path_output_results.mkdir(parents=True, exist_ok=True)

        path_output_hyperparameters.mkdir(parents=True, exist_ok=True)

        #SE DEFINE EL TUNER
        if pretext_task == 'Shuffle' and model_name == 'CONV3D':
            hypermodel = HyperModels.HyperModelShuffleCONV3D(input_shape=(n_frames, dim[0], dim[1], 3), num_classes=num_classes)

        if tuner_type == 'Random_Search':

            if pretext_task == 'Shuffle':

                tuner = Tuners.TunerRandomShuffle(
                    hypermodel,
                    objective=config['Keras_Tuner']['tuner']['objetive'],
                    seed=config['Keras_Tuner']['tuner']['seed'],
                    max_trials=config['Keras_Tuner']['tuner']['max_trials'],
                    executions_per_trial=config['Keras_Tuner']['tuner']['executions_per_trial'],
                    directory=path_output_results,
                    project_name=project_name,
                    overwrite=False
                )
                
        elif tuner_type == 'HyperBand':

            if pretext_task == 'Shuffle':

                tuner = Tuners.TunerHyperBandShuffle(
                    hypermodel,
                    objective=config['Keras_Tuner']['tuner']['objetive'],
                    seed=config['Keras_Tuner']['tuner']['seed'],
                    max_epochs=config['Keras_Tuner']['tuner']['max_epochs'],
                    executions_per_trial=config['Keras_Tuner']['tuner']['executions_per_trial'],
                    directory=path_output_results,
                    project_name=project_name,
                    overwrite=False
                )
                
        else:

            if pretext_task == 'Shuffle':

                tuner = Tuners.TunerBayesianShuffle(
                    hypermodel,
                    objective=config['Keras_Tuner']['tuner']['objetive'],
                    seed=config['Keras_Tuner']['tuner']['seed'],
                    max_trials=config['Keras_Tuner']['tuner']['max_trials'],
                    num_initial_points=config['Keras_Tuner']['tuner']['num_initial_points'],
                    directory=path_output_results,
                    project_name=project_name,
                    overwrite=False
                )

        #SE LLEVA A CABO LA BUSQUEDA DE LOS MEJORES HIPERPARÁMETROS
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

        tuner.search_space_summary()

        start_time = time.time()

        tuner.search(train_ids_instances, validation_ids_instances, dim, path_instances, n_frames, 1, epochs, [earlystopping])

        stop_time = time.time()

        elapsed_time = stop_time - start_time 

        tuner.results_summary()

        best_hp = tuner.get_best_hyperparameters()[0].values

        #Se almacena el tuner en un fichero binario
        with (path_output_results / project_name / 'tuner.pkl').open('wb') as file_descriptor:
            pickle.dump(tuner, file_descriptor)

        with (path_output_results / project_name / 'search_time.txt').open('w') as filehandle:
            filehandle.write("Tiempo de busqueda: %f\n" % elapsed_time)

        #Se guardan los hiperparámetros
        with (path_output_hyperparameters / (project_name + '.json')).open('w') as filehandle:
            json.dump(best_hp, filehandle)
                
    else:

        path_output_results_CL = Path(join(config['Keras_Tuner']['path_dir_results'], dataset, 'Transfer_Learning', pretext_task, tuner_type, model_name, 'Classification_Layer'))

        path_output_results_FT = Path(join(config['Keras_Tuner']['path_dir_results'], dataset, 'Transfer_Learning', pretext_task, tuner_type, model_name, 'Fine_Tuning'))

        path_output_hyperparameters_CL = Path(join(config['Keras_Tuner']['path_hyperparameters'], dataset, 'Transfer_Learning', pretext_task, tuner_type, model_name, 'Classification_Layer'))

        path_output_hyperparameters_FT = Path(join(config['Keras_Tuner']['path_hyperparameters'], dataset, 'Transfer_Learning', pretext_task, tuner_type, model_name, 'Fine_Tuning'))

        path_output_results_CL.mkdir(parents=True, exist_ok=True)

        path_output_results_FT.mkdir(parents=True, exist_ok=True)

        path_output_hyperparameters_CL.mkdir(parents=True, exist_ok=True)

        path_output_hyperparameters_FT.mkdir(parents=True, exist_ok=True)

        path_weights = config['Keras_Tuner']['Transfer_Learning']['path_weights_conv_layers']

        if pretext_task == 'Shuffle' and model_name == 'CONV3D' and not classification:

            hypermodel_cl = HyperModels.HyperModel_FINAL_Shuffle_CONV3D_Regression_CL(input_shape=(n_frames, dim[0], dim[1], 3), num_classes=num_classes, path_weights=path_weights)

        if tuner_type == 'Random_Search':

            if not classification:

                tuner = Tuners.TunerRandomFINALRegression(
                    hypermodel_cl,
                    objective=config['Keras_Tuner']['tuner']['objetive'],
                    seed=config['Keras_Tuner']['tuner']['seed'],
                    max_trials=config['Keras_Tuner']['tuner']['max_trials'],
                    executions_per_trial=config['Keras_Tuner']['tuner']['executions_per_trial'],
                    directory=path_output_results_CL,
                    project_name=project_name,
                    overwrite=False
                )
                
        elif tuner_type == 'HyperBand':

            if not classification:

                tuner = Tuners.TunerHyperBandFINALRegression(
                    hypermodel_cl,
                    objective=config['Keras_Tuner']['tuner']['objetive'],
                    seed=config['Keras_Tuner']['tuner']['seed'],
                    max_epochs=config['Keras_Tuner']['tuner']['max_epochs'],
                    executions_per_trial=config['Keras_Tuner']['tuner']['executions_per_trial'],
                    directory=path_output_results_CL,
                    project_name=project_name,
                    overwrite=False
                )
                
        else:

            if not classification:

                tuner = Tuners.TunerBayesianFINALRegression(
                    hypermodel_cl,
                    objective=config['Keras_Tuner']['tuner']['objetive'],
                    seed=config['Keras_Tuner']['tuner']['seed'],
                    max_trials=config['Keras_Tuner']['tuner']['max_trials'],
                    num_initial_points=config['Keras_Tuner']['tuner']['num_initial_points'],
                    directory=path_output_results_CL,
                    project_name=project_name,
                    overwrite=False
                )

        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

        tuner.search_space_summary()

        start_time = time.time()

        tuner.search(train_ids_instances, validation_ids_instances, dim, path_instances, n_frames, 1, epochs, [earlystopping])

        stop_time = time.time()

        elapsed_time = stop_time - start_time 

        tuner.results_summary()

        best_hp = tuner.get_best_hyperparameters()[0].values

        #Se almacena el tuner en un fichero binario
        with (path_output_results_CL / project_name / 'tuner.pkl').open('wb') as file_descriptor:
            pickle.dump(tuner, file_descriptor)

        with (path_output_results_CL / project_name / 'search_time.txt').open('w') as filehandle:
            filehandle.write("Tiempo de busqueda: %f\n" % elapsed_time)

        #Se guardan los hiperparámetros
        with (path_output_hyperparameters_CL / (project_name + '.json')).open('w') as filehandle:
            json.dump(best_hp, filehandle)


        #Fine Tuning

        if pretext_task == 'Shuffle' and model_name == 'CONV3D' and not classification:
            hypermodel_ft = HyperModels.HyperModel_FINAL_Shuffle_CONV3D_Regression_FT(input_shape=(n_frames, dim[0], dim[1], 3), num_classes=num_classes, path_weights=path_weights, hyperparameters=best_hp)

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
                hypermodel_cl,
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
                hypermodel_cl,
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
            'n_clases': num_classes,
            'n_channels': 3,
            'n_frames': n_frames,
            'normalized': best_hp['normalized'],
            'shuffle': best_hp['normalized'],
        }

        if not classification:
            train_generator = Datagenerators.DataGeneratorFINALRegression(train_ids_instances, **params)
            validation_generator = Datagenerators.DataGeneratorFINALRegression(validation_ids_instances, **params)

        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

        tuner.search_space_summary()

        start_time = time.time()

        tuner.search(x=train_generator, validation_data=validation_generator, epochs=epochs, callbacks=[earlystopping])

        stop_time = time.time()

        elapsed_time = stop_time - start_time 

        tuner.results_summary()

        best_hp = tuner.get_best_hyperparameters()[0].values

        #Se almacena el tuner en un fichero binario
        with (path_output_results_FT / project_name / 'tuner.pkl').open('wb') as file_descriptor:
            pickle.dump(tuner, file_descriptor)

        with (path_output_results_FT / project_name / 'search_time.txt').open('w') as filehandle:
            filehandle.write("Tiempo de busqueda: %f\n" % elapsed_time)

        #Se guardan los hiperparámetros
        with (path_output_hyperparameters_FT / (project_name + '.json')).open('w') as filehandle:
            json.dump(best_hp, filehandle)


run_hyperparameter_tuning()
