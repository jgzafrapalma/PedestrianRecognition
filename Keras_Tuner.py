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

    path_instances = Path(join(config['Keras_Tuner']['path_instances'], dataset))
    path_id_instances = Path(join(config['Keras_Tuner']['path_id_instances'], dataset))

    #Se cargan los identificadores correspondientes a las instancias de entrenamiento y validación
    train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')
    validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')
    #test_ids_instances = read_instance_file_txt(path_id_instances / 'test.txt')

    dim = config['Keras_Tuner']['dim']
    epochs = config['Keras_Tuner']['epochs']
    n_frames = config['Keras_Tuner']['n_frames']
    num_classes = config['Keras_Tuner']['num_classes']

    #Se crean los directorios de salida de los resultados
    path_dir_results = Path(config['Keras_Tuner']['path_dir_results'])
    path_results_dataset = Path(path_dir_results / dataset)
    path_results_pretext_task = Path(path_results_dataset / pretext_task)

    path_dir_hyperparameters = Path(config['Keras_Tuner']['path_hyperparameters'])

    path_hyperparameters_dataset = Path(path_dir_hyperparameters / dataset)

    path_hyperparameters_pretex_task = Path(path_hyperparameters_dataset / pretext_task)


    #Se crean las carpetas necesarias para almacenar los resultados
    if not path_dir_results.exists():
        
        path_dir_results.mkdir()

    if not path_results_dataset.exists():

        path_results_dataset.mkdir()

    if not path_results_pretext_task.exists():

        path_results_pretext_task.mkdir()

    #Se creean las carpetas necesarias para almacenar los hipeparámetros
    if not path_dir_hyperparameters.exists():
        path_dir_hyperparameters.mkdir()

    if not path_hyperparameters_dataset.exists():
        path_hyperparameters_dataset.mkdir()

    if not path_hyperparameters_pretex_task.exists():
        path_hyperparameters_pretex_task.mkdir()


    tuners, paths_results, paths_hyperparameters = define_tuners(
        (n_frames, dim[0], dim[1], 3),
        num_classes,
        pretext_task,
        model_name,
        path_results_pretext_task,
        path_hyperparameters_pretex_task
    )

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

    for id_tuner, tuner in enumerate(tuners):

        tuner.search_space_summary()

        start_time = time.time()

        tuner.search(train_ids_instances, validation_ids_instances, dim, path_instances, n_frames, 1, epochs, [earlystopping])

        stop_time = time.time()

        elapsed_time = stop_time - start_time 

        tuner.results_summary()

        #Para obtener el accuracy en el conjunto de test
        best_model = tuner.get_best_models(num_models=1)[0]

        best_hp = tuner.get_best_hyperparameters()[0].values

        if pretext_task == 'Shuffle':

            params = {
                'dim': dim,
                'path_instances': path_instances,
                'batch_size': best_hp['batch_size'],
                'n_clases': 2,
                'n_channels': 3,
                'n_frames': n_frames,
                'normalized': best_hp['normalized'],
                'shuffle': best_hp['shuffle'],
                'step_swaps': best_hp['step_swaps']
            }

            #test_generator = DataGenerators.DataGeneratorShuffle(test_ids_instances, **params)

            validation_generator = DataGenerators.DataGeneratorShuffle(validation_ids_instances, **params)

        #loss_test, accuracy_test = best_model.evaluate(test_generator)

        loss_validation, accuracy_validation = best_model.evaluate(validation_generator)

        #Se almacena el tuner en un fichero binario
        with (paths_results[id_tuner] / 'tuner.pkl').open('wb') as file_descriptor:
            pickle.dump(tuner, file_descriptor)

        with (paths_results[id_tuner] / 'results.txt').open('w') as filehandle:
            filehandle.write("Tiempo de busqueda: %f\n" % elapsed_time)
            #filehandle.write("Resultados en el conjunto de test: \n")
            #filehandle.write("Loss test: %f\n" % loss_test)
            #filehandle.write("Accuracy test: %f\n" % accuracy_test)
            filehandle.write("Resultados en el conjunto de validación: \n")
            filehandle.write("Loss validation: %f\n" % loss_validation)
            filehandle.write("Accuracy validation: %f\n" % accuracy_validation)

        #Se guardan los hiperparámetros
        with (paths_hyperparameters[id_tuner]).open('w') as filehandle:
            json.dump(best_hp, filehandle)


def define_tuners(input_shape, num_classes, pretext_task, model_name, path_results_pretext_task, path_hyperparameters_pretex_task):
    #El hipermodelo depende de la tarea de pretexto que se esta realizando y del modelo
    if pretext_task == 'Shuffle' and model_name == 'CONV3D':
        hypermodel = HyperModels.HyperModelShuffleCONV3D(input_shape=input_shape, num_classes=num_classes)

    tuners = []
    paths_results = []
    paths_hyperparameters = []

    #Se recorren todos los keras tuner que se encuentren en el fichero de configuración
    for tuner_id in list(config['Keras_Tuner']['tuners']):
        #Se obtiene el tipo del keras tuner
        tuner_type = config['Keras_Tuner']['tuners'][tuner_id]['type']

        path_results_pretext_task_tunertype = Path(path_results_pretext_task / tuner_type)

        if not path_results_pretext_task_tunertype.exists():
            path_results_pretext_task_tunertype.mkdir()

        path_results_pretext_task_tunertype_model = Path(path_results_pretext_task_tunertype / model_name)

        path_hyperparameters_pretex_task_tuner_type = Path(path_hyperparameters_pretex_task / tuner_type)

        if not path_hyperparameters_pretex_task_tuner_type.exists():
            path_hyperparameters_pretex_task_tuner_type.mkdir()

        path_hyperparameters_pretex_task_tuner_type_model = Path(path_hyperparameters_pretex_task_tuner_type / model_name)

        if not path_hyperparameters_pretex_task_tuner_type_model.exists():
            path_hyperparameters_pretex_task_tuner_type_model.mkdir()

        paths_results.append(Path(path_results_pretext_task_tunertype_model / config['Keras_Tuner']['tuners'][tuner_id]['project_name']))

        paths_hyperparameters.append(Path(path_hyperparameters_pretex_task_tuner_type_model / (config['Keras_Tuner']['tuners'][tuner_id]['project_name'] + '.json')))

        if tuner_type == 'Random_Search':

            if pretext_task == 'Shuffle':

                tuners.append(
                    Tuners.TunerRandomShuffle(
                        hypermodel,
                        objective=config['Keras_Tuner']['tuners'][tuner_id]['objetive'],
                        seed=config['Keras_Tuner']['tuners'][tuner_id]['seed'],
                        max_trials=config['Keras_Tuner']['tuners'][tuner_id]['max_trials'],
                        executions_per_trial=config['Keras_Tuner']['tuners'][tuner_id]['executions_per_trial'],
                        directory=path_results_pretext_task_tunertype_model,
                        project_name=config['Keras_Tuner']['tuners'][tuner_id]['project_name'],
                        overwrite=False
                    )
                )

        elif tuner_type == 'HyperBand':

            if pretext_task == 'Shuffle':

                tuners.append(
                    Tuners.TunerHyperBandShuffle(
                        hypermodel,
                        objective=config['Keras_Tuner']['tuners'][tuner_id]['objetive'],
                        seed=config['Keras_Tuner']['tuners'][tuner_id]['seed'],
                        max_epochs=config['Keras_Tuner']['tuners'][tuner_id]['max_epochs'],
                        executions_per_trial=config['Keras_Tuner']['tuners'][tuner_id]['executions_per_trial'],
                        directory=path_results_pretext_task_tunertype_model,
                        project_name=config['Keras_Tuner']['tuners'][tuner_id]['project_name'],
                        overwrite=False
                    )
                )

        else:

            if pretext_task == 'Shuffle':

                tuners.append(
                    Tuners.TunerBayesianShuffle(
                        hypermodel,
                        objective=config['Keras_Tuner']['tuners'][tuner_id]['objetive'],
                        seed=config['Keras_Tuner']['tuners'][tuner_id]['seed'],
                        max_trials=config['Keras_Tuner']['tuners'][tuner_id]['max_trials'],
                        num_initial_points=config['Keras_Tuner']['tuners'][tuner_id]['num_initial_points'],
                        directory=path_results_pretext_task_tunertype_model,
                        project_name=config['Keras_Tuner']['tuners'][tuner_id]['project_name'],
                        overwrite=False
                    )
                )

    return tuners, paths_results, paths_hyperparameters

run_hyperparameter_tuning()
