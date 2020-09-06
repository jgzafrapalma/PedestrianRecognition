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
from pathlib import Path
from os.path import join

dataset = config['Transfer_Learning']['dataset']
pretext_task = config['Transfer_Learning']['pretext_task']
model_name = config['Transfer_Learning']['model_name']
input_model = config['Transfer_Learning']['input_model']

dim = config['Transfer_Learning']['dim']
n_frames = config['Transfer_Learning']['n_frames']
channels = config['Transfer_Learning']['channels']


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

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

model.summary()

