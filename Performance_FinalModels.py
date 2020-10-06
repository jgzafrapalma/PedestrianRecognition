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

if not config['Performance_FinalModels']['random']:

    SEED = config['Performance_FinalModels']['seed']
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


import DataGenerators

from FuncionesAuxiliares import read_instance_file_txt
from os.path import join
from pathlib import Path

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix




#Cargar las variables necesarias del fichero de configuración
dim = config['Performance_FinalModels']['dim']
dataset = config['Performance_FinalModels']['dataset']
type_model = config['Performance_FinalModels']['type_model']
pretext_task = config['Performance_FinalModels']['pretext_task']
model_name = config['Performance_FinalModels']['model_name']
input_model = config['Performance_FinalModels']['input_model']
#Ruta donde se encuentran las instancias que van a ser utilizadas para obtener las predicciones del modelo final
path_instances = Path(join(config['Performance_FinalModels']['path_instances'], dataset, type_model, pretext_task))
path_ids_instances = Path(join(config['Performance_FinalModels']['path_id_instances'], dataset))

#Variables utilizadas por el DataGenerator
n_frames = config['Performance_FinalModels']['n_frames']
batch_size = config['Performance_FinalModels']['batch_size']
n_clases = config['Performance_FinalModels']['n_clases']
normalized = config['Performance_FinalModels']['normalized']
shuffle = config['Performance_FinalModels']['shuffle']


validation_ids_instances = read_instance_file_txt(path_ids_instances / 'validation.txt')

params = {'dim': dim,
          'path_instances': path_instances,
          'batch_size': batch_size,
          'n_clases': n_clases,
          'n_frames': n_frames,
          'n_channels': 3,
          'normalized': normalized,
          'Shuffle': shuffle}

#Se utiliza el DataGenerator correspondiente a la tarea de pretexto
if type_model == 'Crossing-Detection':
    """Inicialización del DataGenerator, en el constructor se inicializa el orden en el que se van a devolver las instancias del problema."""
    validation_generator = DataGenerators.DataGeneratorFINALCrossingDetection(validation_ids_instances, **params)

#Ruta en la que se encuentra el modelo del que se va a evaluar si rendimiento
path_model = Path(join(config['Performance_FinalModels']['path_model'], dataset, type_model, model_name, input_model))

#Se carga el modelo final
model = load_model(path_model)

"""Se obtiene los identificadores de las intancias y su etiqueta en el orden en el que son insertadas en el modelo final"""
id_instances_validation, y_validation = validation_generator.get_ID_instances_and_real_labels()

y_predictions = model.predict(validation_generator)

print(confusion_matrix(y_validation, y_predictions))

with open('predictions.txt', 'w') as filehandle:
    for id_instance, y_real, y_pred in zip(id_instances_validation, y_validation, y_predictions):
        filehandle.write("%s %f %f\n" % (id_instance, y_real, y_pred))