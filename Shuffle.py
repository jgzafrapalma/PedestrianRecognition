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

if not config['Shuffle']['random']:

    SEED = config['Shuffle']['seed']
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

import models
from DataGenerators import DataGeneratorShuffle
from pathlib import Path
from os.path import join
import json
from datetime import datetime
import numpy as np

from FuncionesAuxiliares import read_instance_file_txt

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

date_time = datetime.now().strftime("%d%m%Y-%H%M%S")

#Se cargan las variables necesarias del fichero de configuración
dim = config['Shuffle']['dim']
dataset = config['Shuffle']['dataset']
model_name = config['Shuffle']['model_name']
path_instances = Path(join(config['Shuffle']['path_instances'], dataset))
path_id_instances = Path(join(config['Shuffle']['path_id_instances'], dataset))
epochs = config['Shuffle']['epochs']
n_frames = config['Shuffle']['n_frames']

tensorboard_logs = str(Path(join(config['Shuffle']['tensorboard_logs'], dataset, 'Shuffle', model_name, date_time)))

path_hyperparameters = Path(config['Shuffle']['path_hyperparameters'])

with path_hyperparameters.open('r') as file_descriptor:
    hyperparameters = json.load(file_descriptor)

batch_size = hyperparameters['batch_size']
dense_activation = hyperparameters['dense_activation']
dropout_rate_1 = hyperparameters['dropout_rate_1']
dropout_rate_2 = hyperparameters['dropout_rate_2']
learning_rate = hyperparameters['learning_rate']
normalized = hyperparameters['normalized']
shuffle = hyperparameters['shuffle']
step_swaps = hyperparameters['step_swaps']
unit = hyperparameters['unit']

params = {'dim': dim,
          'path_instances': path_instances,
          'batch_size': batch_size,
          'n_clases': 2,
          'n_channels': 3,
          'n_frames': n_frames,
          'normalized': normalized,
          'shuffle': shuffle,
          'step_swaps': step_swaps}


train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')

validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')

train_generator = DataGeneratorShuffle(train_ids_instances, **params)

validation_generator = DataGeneratorShuffle(validation_ids_instances, **params)

if config['Shuffle']['model_name'] == 'CONV3D':

    model = models.model_Shuffle_CONV3D((n_frames, dim[0], dim[1], 3), dropout_rate_1, dropout_rate_2, dense_activation, unit, learning_rate)

#CALLBACKS

tensorboard = TensorBoard(log_dir=tensorboard_logs, histogram_freq=1, write_images=True)

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

keras_callbacks = [tensorboard, earlystopping, reducelronplateau]

#ENTRENAMIENTO

history = model.fit(x=train_generator, validation_data=validation_generator, epochs=epochs, callbacks=keras_callbacks)


#ALMACENAR LOS RESULTADOS OBTENIDOS DEL ENTRENAMIENTO
path_output_model = Path(join(config['Shuffle']['path_output_model'], dataset, 'Shuffle', model_name, date_time))

#Se crean los directorios en los que se van a almacenar los resultados
path_output_model.mkdir(parents=True, exist_ok=True)

np.save(path_output_model / 'history.npy', history.history)

model.save(path_output_model / 'model.h5')

model.save_weights(str(path_output_model / 'weights.h5'))