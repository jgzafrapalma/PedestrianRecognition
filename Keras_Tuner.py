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

from datetime import datetime

from os.path import join

import time

import DataGenerators

from pathlib import Path

def run_hyperparameter_tuning():

    dataset = config['Keras_Tuner']['dataset']

    pretext_task = config['Keras_Tuner']['pretext_task']
    model_name = config['Keras_Tuner']['model']

    path_instances = Path(join(config['Keras_Tuner']['path_instances'], dataset))
    path_id_instances = Path(join(config['Keras_Tuner']['path_id_instances'], dataset))

    #Se cargan los identificadores correspondientes a las instancias de entrenamiento y validación
    train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')
    validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')
    test_ids_instances = read_instance_file_txt(path_id_instances / 'test.txt')

    dim = config['Keras_Tuner']['dim']
    epochs = config['Keras_Tuner']['epochs']
    n_frames = config['Keras_Tuner']['n_frames']
    num_classes = config['Keras_Tuner']['num_classes']

    #Se crean los directorios de salida de los resultados
    path_dir_results = Path(config['Keras_Tuner']['path_dir_results'])
    path_results_dataset = Path(path_dir_results / dataset)
    path_results_tuners = Path(path_results_dataset / 'tuners')
    path_resultstuners_pretexttask = Path(path_results_tuners / pretext_task)


    path_results_tuners = Path(path_results_dataset / 'tuners_results')

    #Se crean las carpetas necesarias para almacenar los resultados
    if not path_dir_results.exists():
        
        path_dir_results.mkdir()

    if not path_results_dataset.exists():

        path_results_dataset.mkdir()

    if not path_results_hypertuners.exists():

        path_results_hypertuners.mkdir()


        path_resultshypertuners_pretexttask.mkdir()
        path_results_tuners.mkdir()

    tuners, tuners_types, project_names = define_tuners(
        (n_frames, dim[0], dim[1], 3),
        num_classes,
        pretext_task,
        model_name,
        path_resultstuners_pretexttask,
        date_time
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

            test_generator = DataGenerators.DataGeneratorShuffle(test_ids_instances, **params)

            loss_test, accuracy_test = best_model.evaluate(test_generator)

            validation_generator = DataGenerators.DataGeneratorShuffle(validation_ids_instances, **params)

        loss_validation, accuracy_validation = best_model.evaluate(validation_generator)

        date_time = datetime.now().strftime("%m%d%Y-%H%M%S")

        path_results_type = Path(path_results_tuners / tuners_types[id_tuner])

        if not path_results_type.exists():
            path_results_type.mkdir()
        
        path_results_proyect = Path(path_results_type / project_names[id_tuner])

        if not path_results_proyect.exists():
            path_results_proyect.mkdir()

        path_output = Path(path_results_proyect / (date_time + '-' + str(accuracy_test)))

        if not path_output.exists():
            path_output.mkdir()

        #Se almacena el tuner en un fichero binario
        with (path_output / 'tuner.pkl').open('wb') as file_descriptor:
            pickle.dump(tuner, file_descriptor)

        #Se almacena el modelo
        model = tuner.hypermodel.build(best_hp)

        model.save()

        with (path_output / 'results.txt').open('w') as filehandle:
            filehandle.write("Mejores hiperparámetros del mejor modelo:\n")
            filehandle.write(str(best_hp) + "\n")
            filehandle.write("Tiempo de busqueda: %f\n" % elapsed_time)
            filehandle.write("Resultados en el conjunto de test: \n")
            filehandle.write("Loss test: %f\n" % loss_test)
            filehandle.write("Accuracy test: %f\n" % accuracy_test)
            filehandle.write("Resultados en el conjunto de validación: \n")
            filehandle.write("Loss validation: %f\n" % loss_validation)
            filehandle.write("Accuracy validation: %f\n" % accuracy_validation)

def define_tuners(input_shape, num_classes, pretext_task, model_name, path_resultstuners_pretexttask):
    #El hipermodelo depende de la tarea de pretexto que se esta realizando y del modelo
    if pretext_task == 'Shuffle' & model_name == 'CONV3D':
        hypermodel = HyperModels.HyperModelShuffleCONV3D(input_shape=input_shape, num_classes=num_classes)

    tuners = []
    tuners_types = []
    project_names = []

    #Se recorren todos los keras tuner que se encuentren en el fichero de configuración
    for tuner_id in list(config['Keras_Tuner']['tuners']):
        #Se obtiene el tipo del keras tuner
        tuner_type = config['Keras_Tuner']['tuners'][tuner_id]['type']

        path_results_tuners_pretext_task_tunertype = Path(path_resultstuners_pretexttask / tuner_type)

        if not path_results_tuners_pretext_task_tunertype.exists():
            path_results_tuners_pretext_task_tunertype.mkdir()

        path_results_tuners_pretext_task_tunertype_model = Path(path_results_tuners_pretext_task_tunertype / model_name)

        if tuner_type == 'Random_Search':

            if pretext_task == 'Shuffle':

                tuners.append(
                    Tuners.TunerRandomShuffle(
                        hypermodel,
                        objective=config['Keras_Tuner']['tuners'][tuner_id]['objetive'],
                        seed=config['Keras_Tuner']['tuners'][tuner_id]['seed'],
                        max_trials=config['Keras_Tuner']['tuners'][tuner_id]['max_trials'],
                        executions_per_trial=config['Keras_Tuner']['tuners'][tuner_id]['executions_per_trial'],
                        directory=path_results_tuners_pretext_task_tunertype_model,
                        project_name=config['Keras_Tuner']['tuners'][tuner_id]['project_name'],
                        overwrite=False
                    )
                )

            tuners_types.append('Random_Search')

            project_names.append(config['Keras_Tuner']['tuners'][tuner_id]['project_name'])

        elif tuner_type == 'HyperBand':

            if pretext_task == 'Shuffle':

                tuners.append(
                    Tuners.TunerHyperBandShuffle(
                        hypermodel,
                        objective=config['Keras_Tuner']['tuners'][tuner_id]['objetive'],
                        seed=config['Keras_Tuner']['tuners'][tuner_id]['seed'],
                        max_epochs=config['Keras_Tuner']['tuners'][tuner_id]['max_epochs'],
                        executions_per_trial=config['Keras_Tuner']['tuners'][tuner_id]['executions_per_trial'],
                        directory=path_results_tuners_pretext_task_tunertype_model,
                        project_name=config['Keras_Tuner']['tuners'][tuner_id]['project_name'],
                        overwrite=False
                    )
                )

            tuners_types.append('HyperBand')

            project_names.append(config['Keras_Tuner']['tuners'][tuner_id]['project_name'])

        else:

            if pretext_task == 'Shuffle':

                tuners.append(
                    Tuners.TunerBayesianShuffle(
                        hypermodel,
                        objective=config['Keras_Tuner']['tuners'][tuner_id]['objetive'],
                        seed=config['Keras_Tuner']['tuners'][tuner_id]['seed'],
                        max_trials=config['Keras_Tuner']['tuners'][tuner_id]['max_trials'],
                        num_initial_points=config['Keras_Tuner']['tuners'][tuner_id]['num_initial_points'],
                        directory=path_results_tuners_pretext_task_tunertype_model,
                        project_name=config['Keras_Tuner']['tuners'][tuner_id]['project_name'],
                        overwrite=False
                    )
                )

            tuners_types.append('Bayesian_Optimization')

            project_names.append(config['Keras_Tuner']['tuners'][tuner_id]['project_name'])

    return tuners, tuners_types, project_names




run_hyperparameter_tuning()
