#LIMITAR CPU AL 45%
import os, sys
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45

#Se carga el fichero de configuración
import yaml

with open('../../../config.yaml', 'r') as file_descriptor:
    config = yaml.load(file_descriptor, Loader=yaml.FullLoader)


"""Inicialización de los generadores de números aleatorios. Se hace al inicio del codigo para evitar que el importar
otras librerias ya inicializen sus propios generadores"""

if not config['Transfer_Learning_CrossingDetection_Shuffle']['random']:

    SEED = config['Transfer_Learning_CrossingDetection_Shuffle']['seed']
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

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
rootdir = os.path.dirname(parentparentdir)
sys.path.append(os.path.join(rootdir, 'utilities'))


from tensorflow.keras.optimizers import Adam
from pathlib import Path
from os.path import join
import json
import numpy as np
from datetime import datetime

from FuncionesAuxiliares import read_instance_file_txt

import DataGenerators_CrossingDetection_Shuffle, models_CrossingDetection_Shuffle

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau



dataset = config['Transfer_Learning_CrossingDetection_Shuffle']['dataset']
type_model = config['Transfer_Learning_CrossingDetection_Shuffle']['type_model']
project_name = config['Transfer_Learning_CrossingDetection_Shuffle']['project_name']
tuner_type = config['Transfer_Learning_CrossingDetection_Shuffle']['tuner_type']
data_sampling = config['Transfer_Learning_CrossingDetection_Shuffle']['data_sampling']
n_frames = config['Transfer_Learning_CrossingDetection_Shuffle']['n_frames']


path_instances = Path(join(config['Transfer_Learning_CrossingDetection_Shuffle']['path_instances'], dataset, 'CrossingDetection', str(n_frames) + '_frames', data_sampling))
path_id_instances = Path(join(config['Transfer_Learning_CrossingDetection_Shuffle']['path_id_instances'], dataset))

dim = config['Transfer_Learning_CrossingDetection_Shuffle']['dim']
n_classes = config['Transfer_Learning_CrossingDetection_Shuffle']['n_classes']
n_channels = config['Transfer_Learning_CrossingDetection_Shuffle']['n_channels']
epochs = config['Transfer_Learning_CrossingDetection_Shuffle']['epochs']

#date_time = datetime.now().strftime("%d%m%Y-%H%M%S")

tensorboard_logs = str(Path(join(config['Transfer_Learning_CrossingDetection_Shuffle']['tensorboard_logs'], dataset, 'Transfer_Learning', 'CrossingDetection', 'Shuffle', data_sampling, tuner_type, type_model, project_name)))


train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')

validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')


##################################LECTURA DE LOS HIPERPARÁMETROS#######################################
path_hyperparameters_CL = Path(join(config['Transfer_Learning_CrossingDetection_Shuffle']['path_hyperparameters'], dataset, 'Transfer_Learning', 'CrossingDetection', 'Shuffle', tuner_type, type_model, 'Classification_Layer', project_name + '.json'))

path_hyperparameters_FT = Path(join(config['Transfer_Learning_CrossingDetection_Shuffle']['path_hyperparameters'], dataset, 'Transfer_Learning', 'CrossingDetection', 'Shuffle', tuner_type, type_model, 'Fine_Tuning', project_name + '.json'))


with path_hyperparameters_CL.open('r') as file_descriptor:
    hyperparameters_cl = json.load(file_descriptor)

with path_hyperparameters_FT.open('r') as file_descriptor:
    hyperparameters_ft = json.load(file_descriptor)


learning_rate_fine_tuning = hyperparameters_ft['learning_rate']


params = {'dim': dim,
          'path_instances': path_instances,
          'batch_size': hyperparameters_cl['batch_size'],
          'n_clases': n_classes,
          'n_channels': n_channels,
          'n_frames': n_frames,
          'normalized': hyperparameters_cl['normalized'],
          'shuffle': hyperparameters_cl['shuffle']}

train_generator = DataGenerators_CrossingDetection_Shuffle.DataGeneratorCrossingDetectionShuffe(train_ids_instances, **params)

validation_generator = DataGenerators_CrossingDetection_Shuffle.DataGeneratorCrossingDetectionShuffe(validation_ids_instances, **params)


if type_model == 'CONV3D':

    dropout_rate_1 = hyperparameters_cl['dropout_rate_1']
    dropout_rate_2 = hyperparameters_cl['dropout_rate_2']
    dense_activation = hyperparameters_cl['dense_activation']
    unit = hyperparameters_cl['unit']
    learning_rate = hyperparameters_cl['learning_rate']

    #El modelo es definido con las capas convolucionales congeladas
    model = models_CrossingDetection_Shuffle.model_CrossingDetection_Shuffle_CONV3D((n_frames, dim[0], dim[1], n_channels), dropout_rate_1, dropout_rate_2, dense_activation, unit, learning_rate)


elif type_model == 'C3D':

    dropout_rate_1 = hyperparameters_cl['dropout_rate_1']
    dropout_rate_2 = hyperparameters_cl['dropout_rate_2']
    units_dense_layers_1 = hyperparameters_cl['units_dense_layers_1']
    units_dense_layers_2 = hyperparameters_cl['units_dense_layers_2']
    learning_rate = hyperparameters_cl['learning_rate']

    model = models_CrossingDetection_Shuffle.model_CrossingDetection_Shuffle_C3D((n_frames, dim[0], dim[1], n_channels), dropout_rate_1, dropout_rate_2, units_dense_layers_1, units_dense_layers_2, learning_rate)

    

#######################################################################################################

"""SE DEFINE EL NUEVO MODELO Y SE CARGAN LOS PESOS DE LAS CAPAS CONVOLUCIONES APRENDIDOS A TRAVÉS DE
LA TAREA DE PRETEXTO"""
path_weights = Path(join(config['Transfer_Learning_CrossingDetection_Shuffle']['path_weights'], dataset, 'Shuffle', data_sampling, tuner_type, type_model, project_name, 'weights.h5'))

"""En vez de cargar el modelo se van a cargar los pesos sobre un nuevo modelo generado, en el que
los pesos solo van a ser cargados en las capas de convolución"""

model.load_weights(str(path_weights), by_name=True)

model.summary()

#CALLBACKS

tensorboard = TensorBoard(log_dir=tensorboard_logs, histogram_freq=1, write_images=True)

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

keras_callbacks = [tensorboard, earlystopping, reducelronplateau]

#ENTRENAMIENTO

model.fit(x=train_generator, validation_data=validation_generator, epochs=epochs, callbacks=keras_callbacks)

#FINE TUNING

#SE DESCONGELAN TODAS LAS CAPAS PARA REALIZAR EL AJUSTE FINO

model.trainable = True

"""Se vuelve a realizar un entrenamiento pero ahora modificando los pesos de todas las capas, con un
coeficiente de aprendizaje bajo (obtenido a partir de optimizando de hiperparámetros)"""
model.compile(optimizer=Adam(learning_rate=learning_rate_fine_tuning), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(x=train_generator, validation_data=validation_generator, epochs=epochs, callbacks=keras_callbacks)


#GUARDADO DEL MODELO FINAL, PESOS Y HISTORY

path_output_model = Path(join(config['Transfer_Learning_CrossingDetection_Shuffle']['path_output_model'], dataset, 'Transfer_Learning', 'CrossingDetection', 'Shuffle', data_sampling, tuner_type, type_model, project_name))

#Se crean los directorios en los que se van a almacenar los resultados
path_output_model.mkdir(parents=True, exist_ok=True)

np.save(path_output_model / 'history.npy', history.history)

#model.save(path_output_model / 'model.h5')

model.save_weights(str(path_output_model / 'weights.h5'))
