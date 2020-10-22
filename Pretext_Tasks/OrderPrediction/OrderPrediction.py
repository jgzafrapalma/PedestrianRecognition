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
rootdir = os.path.dirname(parentdir)

with open(os.path.join(rootdir, 'config.yaml'), 'r') as file_descriptor:
    config = yaml.load(file_descriptor, Loader=yaml.FullLoader)


"""Inicialización de los generadores de números aleatorios. Se hace al inicio del codigo para evitar que el importar
otras librerias ya inicializen sus propios generadores"""

if not config['OrderPrediction']['random']:

    SEED = config['OrderPrediction']['seed']
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

#Se añade el directorio utilities a sys.path para que pueda ser usaado por el comando import
sys.path.append(os.path.join(rootdir, 'utilities'))

from os.path import join
import json
import numpy as np
from pathlib import Path
#from datetime import datetime

import DataGenerator_OrderPrediction, models_OrderPrediction

from FuncionesAuxiliares import read_instance_file_txt

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

#date_time = datetime.now().strftime("%d%m%Y-%H%M%S")

#Se cargan las variables necesarias del fichero de configuración
dim = config['OrderPrediction']['dim']
dataset = config['OrderPrediction']['dataset']
type_model = config['OrderPrediction']['type_model']
data_sampling = config['OrderPrediction']['data_sampling']
tuner_type = config['OrderPrediction']['tuner_type']
project_name = config['OrderPrediction']['project_name']


path_instances = Path(join(config['OrderPrediction']['path_instances'], dataset, 'OrderPrediction', data_sampling))
path_id_instances = Path(join(config['OrderPrediction']['path_id_instances'], dataset))


epochs = config['OrderPrediction']['epochs']
n_classes = config['OrderPrediction']['n_classes']
n_channels = config['OrderPrediction']['n_channels']

tensorboard_logs = str(Path(join(config['OrderPrediction']['tensorboard_logs'], dataset, 'OrderPrediction', data_sampling, tuner_type, type_model, project_name)))

#Se carga la ruta en la que se encuentra el fichero con los hiperparámetros
path_hyperparameters = Path(join(config['OrderPrediction']['path_hyperparameters'], dataset, 'OrderPrediction', data_sampling, tuner_type, type_model, project_name + '.json'))

with path_hyperparameters.open('r') as file_descriptor:
    hyperparameters = json.load(file_descriptor)

#Se cargan los hiperparámetros necesarios en el DataGenerator
batch_size = hyperparameters['batch_size']
normalized = hyperparameters['normalized']
shuffle = hyperparameters['shuffle']

params = {'dim': dim,
          'path_instances': path_instances,
          'batch_size': batch_size,
          'n_clases': n_classes,
          'n_channels': n_channels,
          'normalized': normalized,
          'shuffle': shuffle,
          'n_epochs': epochs}

train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')

validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')

train_generator = DataGenerator_OrderPrediction.DataGeneratorOrderPrediction(train_ids_instances, **params)

validation_generator = DataGenerator_OrderPrediction.DataGeneratorOrderPrediction(validation_ids_instances, **params)

if type_model == 'SIAMESE':

    units_dense_layers_1 = hyperparameters['units_dense_layers_1']
    units_dense_layers_2 = hyperparameters['units_dense_layers_2']
    learning_rate = hyperparameters['learning_rate']

    model = models_OrderPrediction.model_OrderPrediction_SIAMESE((dim[0], dim[1], n_channels), units_dense_layers_1, units_dense_layers_2, learning_rate)

#CALLBACKS

tensorboard = TensorBoard(log_dir=tensorboard_logs, histogram_freq=1, write_images=True)

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

keras_callbacks = [tensorboard, earlystopping, reducelronplateau]

#ENTRENAMIENTO

history = model.fit(x=train_generator, validation_data=validation_generator, epochs=epochs, callbacks=keras_callbacks)


#ALMACENAR LOS RESULTADOS OBTENIDOS DEL ENTRENAMIENTO
path_output_model = Path(join(config['OrderPrediction']['path_output_model'], dataset, 'OrderPrediction', data_sampling, tuner_type, project_name))

#Se crean los directorios en los que se van a almacenar los resultados
path_output_model.mkdir(parents=True, exist_ok=True)

np.save(path_output_model / 'history.npy', history.history)

#model.save(path_output_model / 'model.h5')

model.save_weights(str(path_output_model / 'weights.h5'))