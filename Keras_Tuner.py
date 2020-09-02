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

from HyperModel import CNNHyperModel
from FuncionesAuxiliares import read_instance_file_txt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

from Tuner import MyTunerBayesian, MyTunerRandom, MyTunerHyperBand

from loguru import logger

import pickle

from os.path import join

from datetime import datetime

import time

from DataGenerator import DataGenerator

def run_hyperparameter_tuning():

    #Se cargan los identificadores correspondientes a las instancias de entrenamiento y validación
    train_ids_instances = read_instance_file_txt(config['Keras_Tuner']['path_train_id_instances'])
    validation_ids_instances = read_instance_file_txt(config['Keras_Tuner']['path_validation_id_instances'])
    test_ids_instances = read_instance_file_txt(config['Keras_Tuner']['path_test_id_instances'])

    dim = config['Keras_Tuner']['dim']
    path_instances = config['Keras_Tuner']['path_instances']
    epochs = config['Keras_Tuner']['epochs']
    n_frames = config['Keras_Tuner']['n_frames']
    num_classes = config['Keras_Tuner']['num_classes']
    path_dir_results = config['Keras_Tuner']['path_dir_results']

    tuners, tuners_types, project_names = define_tuners(
        (n_frames, dim[0], dim[1], 3),
        num_classes,
    )

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

    for id_tuner, tuner in enumerate(tuners):

        tuner.search_space_summary()

        logger.info(f"Iniciando busqueda para {tuner}")

        start_time = time.time()

        tuner.search(train_ids_instances, validation_ids_instances, dim, path_instances, n_frames, 1, epochs, [earlystopping])

        stop_time = time.time()

        logger.success(f"Busqueda para {tuner} finalizada")

        elapsed_time = start_time - stop_time

        tuner.results_summary()

        best_model = tuner.get_best_models(num_models=1)[0]

        best_hp = tuner.get_best_hyperparameters()[0].values

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

        test_generator = DataGenerator(test_ids_instances, **params)

        loss, accuracy = best_model.evaluate(test_generator)

        date_time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")

        path_output = join(path_dir_results, tuners_types[id_tuner], project_names[id_tuner], f"{date_time}-{accuracy}")

        #Se almacena el tuner en un fichero binario
        with open(join(path_output, 'tuner'), 'wb') as file_descriptor:
            pickle.dump(tuner, file_descriptor)

        logger.success(f"Almacenado fichero binario para {tuner}")

        with open(join(path_output, "results.txt"), 'w') as filehandle:
            filehandle.write("Mejores hiperparámetros del mejor modelo:\n")
            filehandle.write(best_hp + "\n")
            filehandle.write("Tiempo de busqueda: " + elapsed_time + "\n")
            filehandle.write('Resultados de los modelos para el conjunto de validación: \n')
            filehandle.write(tuner.results_summary())

        logger.success(f"Almacenados resultados en fichero de texto")


def define_tuners(input_shape, num_classes):

    #Instanciación del objeto de la clase HyperModel
    hypermodel = CNNHyperModel(input_shape=input_shape, num_classes=num_classes)

    tuners = []
    tuners_types = []
    project_names = []

    #Se recorren todos los keras tuner que se encuentren en el fichero de configuración
    for tuner_id in list(config['Keras_Tuner']['tuners']):
        #Se obtiene el tipo del keras tuner
        tuner_type = config['Keras_Tuner']['tuners'][tuner_id]['type']

        if tuner_type == 'Random_Search':

            tuners.append(
                MyTunerRandom(
                    hypermodel,
                    objective=config['Keras_Tuner']['tuners'][tuner_id]['objetive'],
                    seed=config['Keras_Tuner']['tuners'][tuner_id]['seed'],
                    max_trials=config['Keras_Tuner']['tuners'][tuner_id]['max_trials'],
                    executions_per_trial=config['Keras_Tuner']['tuners'][tuner_id]['executions_per_trial'],
                    directory=config['Keras_Tuner']['tuners'][tuner_id]['directory'],
                    project_name=config['Keras_Tuner']['tuners'][tuner_id]['project_name'],
                    overwrite=True
                )
            )

            tuners_types.append('Random_Search')

            project_names.append(config['Keras_Tuner']['tuners'][tuner_id]['project_name'])

        elif tuner_type == 'HyperBand':

            tuners.append(
                MyTunerHyperBand(
                    hypermodel,
                    objective=config['Keras_Tuner']['tuners'][tuner_id]['objetive'],
                    seed=config['Keras_Tuner']['tuners'][tuner_id]['seed'],
                    max_epochs=config['Keras_Tuner']['tuners'][tuner_id]['max_epochs'],
                    executions_per_trial=config['Keras_Tuner']['tuners'][tuner_id]['executions_per_trial'],
                    directory=config['Keras_Tuner']['tuners'][tuner_id]['directory'],
                    project_name=config['Keras_Tuner']['tuners'][tuner_id]['project_name'],
                    overwrite=True
                )
            )

            tuners_types.append('HyperBand')

            project_names.append(config['Keras_Tuner']['tuners'][tuner_id]['project_name'])

        else:

            tuners.append(
                MyTunerBayesian(
                    hypermodel,
                    objective=config['Keras_Tuner']['tuners'][tuner_id]['objetive'],
                    seed=config['Keras_Tuner']['tuners'][tuner_id]['seed'],
                    max_trials=config['Keras_Tuner']['tuners'][tuner_id]['max_trials'],
                    num_initial_points=config['Keras_Tuner']['tuners'][tuner_id]['num_initial_points'],
                    directory="Bayesian_search",
                    project_name="CONV3D",
                    overwrite=True
                )
            )

            tuners_types.append('Baayesian_Optimization')

            project_names.append(config['Keras_Tuner']['tuners'][tuner_id]['project_name'])

    return tuners, tuners_types, project_names




run_hyperparameter_tuning()