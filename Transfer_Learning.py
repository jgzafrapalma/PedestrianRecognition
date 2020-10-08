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

if not config['Transfer_Learning']['random']:

    SEED = config['Transfer_Learning']['seed']
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

#########################################################################################################################

from tensorflow import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from os.path import join
import json
from FuncionesAuxiliares import read_instance_file_txt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from datetime import datetime

from sklearn.metrics import confusion_matrix

import DataGenerators

import models

import numpy as np

dataset = config['Transfer_Learning']['dataset']
pretext_task = config['Transfer_Learning']['pretext_task']
model_name = config['Transfer_Learning']['model_name']
input_model = config['Transfer_Learning']['input_model']
type_model = config['Transfer_Learning']['type_model']

path_instances = Path(join(config['Transfer_Learning']['path_instances'], dataset))
path_id_instances = Path(join(config['Transfer_Learning']['path_id_instances'], dataset))

dim = config['Transfer_Learning']['dim']
n_frames = config['Transfer_Learning']['n_frames']
n_classes = config['Transfer_Learning']['n_classes']
channels = config['Transfer_Learning']['channels']
epochs = config['Transfer_Learning']['epochs']
batch_size = config['Transfer_Learning']['batch_size']
normalized = config['Transfer_Learning']['normalized']
shuffle = config['Transfer_Learning']['shuffle']


date_time = datetime.now().strftime("%d%m%Y-%H%M%S")

tensorboard_logs = str(Path(join(config['Transfer_Learning']['tensorboard_logs'], dataset, 'Transfer_Learning', pretext_task, model_name, date_time)))

params = {'dim': dim,
          'path_instances': path_instances,
          'batch_size': batch_size,
          'n_clases': n_classes,
          'n_channels': channels,
          'n_frames': n_frames,
          'normalized': normalized,
          'shuffle': shuffle}


train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')

validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')


##################################LECTURA DE LOS HIPERPARÁMETROS#######################################
path_hyperparameters_cl = Path(config['Transfer_Learning']['path_hyperparameters_classification_layer'])
path_hyperparameters_ft = Path(config['Transfer_Learning']['path_hyperparameters_fine_tuning'])

with path_hyperparameters_cl.open('r') as file_descriptor:
    hyperparameters_cl = json.load(file_descriptor)

with path_hyperparameters_ft.open('r') as file_descriptor:
    hyperparameters_ft = json.load(file_descriptor)

if pretext_task == 'Shuffle' and model_name == 'CONV3D':

    dropout_rate_1 = hyperparameters_cl['dropout_rate_1']
    dropout_rate_2 = hyperparameters_cl['dropout_rate_2']
    dense_activation = hyperparameters_cl['dense_activation']
    unit = hyperparameters_cl['unit']
    learning_rate = hyperparameters_cl['learning_rate']

    learning_rate_fine_tuning = hyperparameters_ft['learning_rate']

    if type_model == 'Crossing-detection':

        train_generator = DataGenerators.DataGeneratorFINALCrossingDetection(train_ids_instances, **params)

        validation_generator = DataGenerators.DataGeneratorFINALCrossingDetection(validation_ids_instances, **params)

        #El modelo es definido con las capas convolucionales congeladas
        model = models.model_FINAL_Shuffle_CONV3D_CrossingDetection((n_frames, dim[0], dim[1], channels), dropout_rate_1, dropout_rate_2, dense_activation, unit, learning_rate)

#######################################################################################################

"""SE DEFINE EL NUEVO MODELO Y SE CARGAN LOS PESOS DE LAS CAPAS CONVOLUCIONES APRENDIDOS A TRAVÉS DE
LA TAREA DE PRETEXTO""" 
path_weights = Path(join(config['Transfer_Learning']['path_weights'], dataset, pretext_task, model_name, input_model, 'weights.h5'))

"""En vez de cargar el modelo se van a cargar los pesos sobre un nuevo modelo generado, en el que
los pesos solo van a ser cargados en las capas de convolución"""

model.load_weights(str(path_weights), by_name=True)

model.summary()

#CALLBACKS

tensorboard = TensorBoard(log_dir=tensorboard_logs, histogram_freq=1, write_images=True)

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

#reducelronplateau = ReduceLROnPlateau(monitor='val_mse', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

keras_callbacks = [tensorboard, earlystopping]

#ENTRENAMIENTO

model.fit(x=train_generator, validation_data=validation_generator, epochs=epochs, callbacks=keras_callbacks)

#FINE TUNING

#SE DESCONGELAN TODAS LAS CAPAS PARA REALIZAR EL AJUSTE FINO

model.trainable = True

if type_model == 'Crossing-detection':
    """Se vuelve a realizar un entrenamiento pero ahora modificando los pesos de todas las capas, con un
    coeficiente de aprendizaje bajo (obtenido a partir de optimizando de hiperparámetros)"""
    model.compile(optimizer=Adam(learning_rate=learning_rate_fine_tuning), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x=train_generator, validation_data=validation_generator, epochs=epochs, callbacks=keras_callbacks)


#GUARDADO DEL MODELO FINAL, PESOS Y HISTORY

path_output_model = Path(join(config['Shuffle']['path_output_model'], dataset, 'Transfer_Learning', pretext_task, model_name, date_time))

#Se crean los directorios en los que se van a almacenar los resultados
path_output_model.mkdir(parents=True, exist_ok=True)

np.save(path_output_model / 'history.npy', history.history)

model.save(path_output_model / 'model.h5')

model.save_weights(str(path_output_model / 'weights.h5'))


y_predictions = model.predict(x=validation_generator)

"""Se obtiene los identificadores de las intancias y su etiqueta en el orden en el que son insertadas en el modelo final"""
id_instances_validation, y_validation = validation_generator.get_ID_instances_and_labels()

print(confusion_matrix(y_validation, y_predictions))

with open('predictions.txt', 'w') as filehandle:
    for id_instance, y_real, y_pred in zip(id_instances_validation, y_validation, y_predictions):
        filehandle.write("%s %f %f\n" % (id_instance, y_real, y_pred))