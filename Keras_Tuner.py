"""Inicialización de los generadores de números aleatorios. Se hace al inicio del codigo para evitar que el importar
otras librerias ya inicializen sus propios generadores"""
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
from random import seed
seed(1)
SEED = 1

#############################################SOLUCIONAR EL ERROR DE LA LIBRERIA CUDNN###################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

########################################################################################################################

from HyperModel import CNNHyperModel
from FuncionesAuxiliares import read_instance_file_txt
from DataGenerator import DataGenerator

from tensorflow.keras.callbacks import EarlyStopping

import argparse

from Tuner import MyTunerBayesian

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dim', nargs=2, default=(128, 128), type=int, dest='dim', help='dimensionality of input data')

parser.add_argument('-t', '--train', default='/media/jorge/DATOS/TFG/datasets/ids_instances/train.txt', type=str, dest='path_train_instances', help='path of the train instances')

parser.add_argument('-v', '--validation', default='/media/jorge/DATOS/TFG/datasets/ids_instances/validation.txt', type=str, dest='path_validation_instances', help='path of the validation instances')

parser.add_argument('-e', '--epochs', default=100, type=int, dest='epochs', help='number of epochs')

parser.add_argument('-f', '--frames', default=8, type=int, dest='n_frames', help='number of frames')


args = parser.parse_args()

#VARIABLES GLOBALES DEL PROBLEMA. PONERLAS COMO ENTRADA POR LINEA DE ARGUMENTOS!!!!
NUM_CLASSES = 2
INPUT_SHAPE = (8, 128, 128, 3)

N_EPOCH_SEARCH = args.epochs
HYPERBAND_MAX_EPOCHS = 40
MAX_TRIALS = 100
EXECUTION_PER_TRIAL = 2
BAYESIAN_NUM_INITIAL_POINTS = 5

def run_hyperparameter_tuning():

    #Instanciación del objeto de la clase HyperModel
    hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)


    tuner = MyTunerBayesian(
        hypermodel,
        objective="val_accuracy",
        seed=SEED,
        max_trials=MAX_TRIALS,
        num_initial_points=BAYESIAN_NUM_INITIAL_POINTS,
        directory="Bayesian_search",
        project_name="CONV3D",
        overwrite=True
    )

    tuner.search_space_summary()


    #Se cargan los identificadores correspondientes a las instancias de entrenamiento y validación
    train_ids_instances = read_instance_file_txt(args.path_train_instances)
    validation_ids_instances = read_instance_file_txt(args.path_validation_instances)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min',
                                  restore_best_weights=True)

    keras_callbacks = [earlystopping]

    tuner.search(train_ids_instances, validation_ids_instances, tuple(args.dim), args.path_instances, args.n_frames, 1, N_EPOCH_SEARCH, keras_callbacks)

    tuner.results_summary()

    best_model = tuner.get_best_models(num_models=1)[0]

    best_hp = tuner.get_best_hyperparameters()


def define_tuners():


run_hyperparameter_tuning()