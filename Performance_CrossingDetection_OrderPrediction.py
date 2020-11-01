#LIMITAR CPU AL 45%
import os, sys
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

if not config['Performance_CrossingDetection_OrderPrediction']['random']:

    SEED = config['Performance_CrossingDetection_OrderPrediction']['seed']
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

rootdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(rootdir, 'utilities'))
sys.path.append(os.path.join(rootdir, 'Downstream_Tasks', 'CrossingDetection', 'OrderPrediction'))


import DataGenerators_CrossingDetection_OrderPrediction, models_CrossingDetection_OrderPrediction

from FuncionesAuxiliares import read_instance_file_txt
from os.path import join
from pathlib import Path
import json
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, classification_report


#Cargar las variables necesarias del fichero de configuración
dim = config['Performance_CrossingDetection_OrderPrediction']['dim']
dataset = config['Performance_CrossingDetection_OrderPrediction']['dataset']
type_model = config['Performance_CrossingDetection_OrderPrediction']['type_model']
data_sampling = config['Performance_CrossingDetection_OrderPrediction']['data_sampling']
project_name = config['Performance_CrossingDetection_OrderPrediction']['project_name']
tuner_type = config['Performance_CrossingDetection_OrderPrediction']['tuner_type']
n_channels = config['Performance_CrossingDetection_OrderPrediction']['n_channels']

#Ruta donde se encuentran las instancias que van a ser utilizadas para obtener las predicciones del modelo final
path_instances = Path(join(config['Performance_CrossingDetection_OrderPrediction']['path_instances'], dataset, 'CrossingDetection', '4_frames', data_sampling))
path_ids_instances = Path(join(config['Performance_CrossingDetection_OrderPrediction']['path_id_instances'], dataset))


path_hyperparameters_CL = Path(join(config['Performance_CrossingDetection_OrderPrediction']['path_hyperparameters'], dataset, 'Transfer_Learning', 'CrossingDetection', 'OrderPrediction', tuner_type, type_model, 'Classification_Layer', project_name + '.json'))
path_hyperparameters_FT = Path(join(config['Performance_CrossingDetection_OrderPrediction']['path_hyperparameters'], dataset, 'Transfer_Learning', 'CrossingDetection', 'OrderPrediction', tuner_type, type_model, 'Fine_Tuning', project_name + '.json'))


with path_hyperparameters_CL.open('r') as file_descriptor:
    hyperparameters_cl = json.load(file_descriptor)

with path_hyperparameters_FT.open('r') as file_descriptor:
    hyperparameters_ft = json.load(file_descriptor)


learning_rate_fine_tuning = hyperparameters_ft['learning_rate']


params = {'dim': dim,
          'path_instances': path_instances,
          'batch_size': hyperparameters_cl['batch_size'],
          'n_clases': 2,
          'n_channels': n_channels,
          'normalized': hyperparameters_cl['normalized'],
          'shuffle': hyperparameters_cl['shuffle']}


validation_ids_instances = read_instance_file_txt(path_ids_instances / 'test.txt')

validation_generator = DataGenerators_CrossingDetection_OrderPrediction.DataGeneratorCrossingDetectionOrderPrediction(validation_ids_instances, **params)


if type_model == 'SIAMESE':

    units_dense_layer_1 = hyperparameters_cl['units_dense_layer_1']
    units_dense_layer_2 = hyperparameters_cl['units_dense_layer_2']

    model = models_CrossingDetection_OrderPrediction.model_CrossingDetection_OrderPrediction_SIAMESE((dim[0], dim[1], n_channels), units_dense_layer_1, units_dense_layer_2, learning_rate_fine_tuning)


#Ruta en la que se encuentra el modelo del que se va a evaluar si rendimiento
path_weights = Path(join(config['Performance_CrossingDetection_OrderPrediction']['path_weights'], dataset, 'Transfer_Learning', 'CrossingDetection', 'OrderPrediction', data_sampling, tuner_type, type_model, project_name, 'weights.h5'))

#Se carga el modelo final
model.load_weights(str(path_weights), by_name=True)

y_predictions = model.predict(x=validation_generator)

y_prob_positive = y_predictions[:, 1]

y_predictions = np.round(y_predictions)

"""Se obtiene los identificadores de las intancias y su etiqueta en el orden en el que son insertadas en el modelo final"""
id_instances_validation, y_validation = validation_generator.get_ID_instances_and_labels()

y_true = y_validation.argmax(axis=1)
y_pred = y_predictions.argmax(axis=1)

print("MATRIZ DE CONFUSIÓN: ")
print(confusion_matrix(y_true, y_pred))

print("ACCURACY: %f" % accuracy_score(y_true, y_pred))

print("F1 Score: %f" % f1_score(y_true, y_pred))

print("ROC Score: %f" % roc_auc_score(y_true, y_prob_positive))

print("CLASSIFICATION REPORT: ")

print(classification_report(y_true, y_pred, target_names=['No crossing', 'Crossing']))

#CALCULO DE LA CURVA ROC

fpr, tpr, _ = roc_curve(y_true, y_prob_positive)

plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.savefig('curveRoc.png')


#with open('predictions.txt', 'w') as filehandle:
    #for id_instance, y_real, y_pred in zip(id_instances_validation, y_validation, y_predictions):
        #filehandle.write("%s %f %f\n" % (id_instance, y_real, y_pred))