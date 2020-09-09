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


from tensorflow import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from os.path import join
from FuncionesAuxiliares import read_instance_file_txt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from datetime import datetime

from DataGenerators import DataGeneratorFINALMODEL

dataset = config['Transfer_Learning']['dataset']
pretext_task = config['Transfer_Learning']['pretext_task']
model_name = config['Transfer_Learning']['model_name']
input_model = config['Transfer_Learning']['input_model']

path_instances = Path(join(config['Transfer_Learning']['path_instances'], dataset))
path_id_instances = Path(join(config['Transfer_Learning']['path_id_instances'], dataset))

dim = config['Transfer_Learning']['dim']
n_frames = config['Transfer_Learning']['n_frames']
channels = config['Transfer_Learning']['channels']
epochs = config['Transfer_Learning']['epochs']
batch_size = config['Transfer_Learning']['batch_size']
normalized = config['Transfer_Learning']['normalized']
shuffle = config['Transfer_Learning']['shuffle']


date_time = datetime.now().strftime("%d%m%Y-%H%M%S")

tensorboard_logs = str(Path(join(config['Transfer_Learning']['tensorboard_logs'], dataset, 'Shuffle', model_name, date_time)))

params = {'dim': dim,
          'path_instances': path_instances,
          'batch_size': batch_size,
          'n_clases': 1,
          'n_channels': 3,
          'n_frames': n_frames,
          'normalized': normalized,
          'shuffle': shuffle}


train_ids_instances = read_instance_file_txt(path_id_instances / 'train.txt')

validation_ids_instances = read_instance_file_txt(path_id_instances / 'validation.txt')

train_generator = DataGeneratorFINALMODEL(train_ids_instances, **params)

validation_generator = DataGeneratorFINALMODEL(validation_ids_instances, **params)



#SE CARGA EL MODELO ENTRENADO PREVIAMENTE
path_model = Path(join(config['Transfer_Learning']['path_models'], dataset, pretext_task, model_name, input_model, 'model.h5'))


base_model = keras.models.load_model(path_model)

base_model.summary()

#SE ELIMINAN LAS CAPAS DE CLASIFICACIÓN DEL MODELO

base_model.trainable = False

last_layer_output = base_model.get_layer('conv3d_3').output

x = Dropout(0.5)(last_layer_output)

x = Flatten()(x)

x = Dense(units=512, activation='relu')(x)

x = Dropout(0.5)(x)

outputs = Dense(1, activation='sigmoid')(x)

#SE CONGELAN LAS CAPAS RESTANTES (CAPAS DE CONVOLUCIÓN)



#SE AÑADEN NUEVAS CAPAS DE CLASIFICACIÓN


#inputs = keras.Input(shape=(n_frames, dim[0], dim[1], channels))

#base_model(inputs, training=False)


model = keras.Model(base_model.input, outputs)

#CALLBACKS

#tensorboard = TensorBoard(log_dir=tensorboard_logs, histogram_freq=1, write_images=True)

earlystopping = EarlyStopping(monitor='val_mse', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

#reducelronplateau = ReduceLROnPlateau(monitor='val_mse', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

keras_callbacks = [earlystopping]

#ENTRENAMIENTO

optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

model.summary()

history = model.fit(generator=train_generator, validation_data=validation_generator, epochs=epochs, callbacks=keras_callbacks)

#ALMACENAR EL MODELO OBTENIDO

"""Inicialización del DataGenerator, en el constructor se inicializa el orden en el que se van a devolver las instancias del problema."""
validation_generator = DataGeneratorFINALMODEL(validation_ids_instances, **params)

"""Se obtiene los identificadores de las intancias y se etiqueta en el orden en el que son insertadas en el modelo final"""
id_instances_validation, y_validation = DataGeneratorFINALMODEL.get_ID_instances_and_real_labels()

y_predictions = model.predict(validation_generator)

with open('predictions.txt', 'w') as filehandle:
    filehandle.write(str(y_predictions))


