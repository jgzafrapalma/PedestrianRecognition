#Se carga el fichero de configuración
import yaml

with open('config.yaml', 'r') as file_descriptor:
    config = yaml.load(file_descriptor, Loader=yaml.FullLoader)

"""Inicialización de los generadores de números aleatorios. Se hace al inicio del codigo para evitar que el importar
otras librerias ya inicializen sus propios generadores"""

semilla = config['Keras_Tuner']['seed']

from numpy.random import seed
seed(semilla)
import tensorflow as tf
tf.random.set_seed(semilla)
from random import seed
seed(semilla)
SEED = semilla

#############################################SOLUCIONAR EL ERROR DE LA LIBRERIA CUDNN###################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

########################################################################################################################

from HyperModel import CNNHyperModel
from FuncionesAuxiliares import read_instance_file_txt

from tensorflow.keras.callbacks import EarlyStopping


from Tuner import MyTunerBayesian

#VARIABLES GLOBALES DEL PROBLEMA.
NUM_CLASSES = config['Keras_Tuner']['num_classes']
INPUT_SHAPE = (config['Keras_Tuner']['n_frames'], config['Keras_Tuner']['dim'][0], config['Keras_Tuner']['dim'][1], 3)

N_EPOCH_SEARCH = config['Keras_Tuner']['epochs']
HYPERBAND_MAX_EPOCHS = 40
MAX_TRIALS = 100 # Number of hyperparameter combinations that will be tested by the tuner
EXECUTION_PER_TRIAL = 2
BAYESIAN_NUM_INITIAL_POINTS = 5

def run_hyperparameter_tuning():

    #Se cargan los identificadores correspondientes a las instancias de entrenamiento y validación
    train_ids_instances = read_instance_file_txt(config['Keras_Tuner']['path_train_instances'])
    validation_ids_instances = read_instance_file_txt(config['Keras_Tuner']['path_validation_instances'])

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


    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min',
                                  restore_best_weights=True)

    keras_callbacks = [earlystopping]

    tuner.search(train_ids_instances, validation_ids_instances, tuple(args.dim), args.path_instances, args.n_frames, 1, N_EPOCH_SEARCH, keras_callbacks)

    tuner.results_summary()

    best_model = tuner.get_best_models(num_models=1)[0]

    best_hp = tuner.get_best_hyperparameters()


def define_tuners():


run_hyperparameter_tuning()