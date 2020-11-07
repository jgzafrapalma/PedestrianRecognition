#LIMITAR CPU AL 45%
import os, sys
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45

#Se carga el fichero de configuración
import yaml

currentdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.dirname(currentdir)

with open(os.path.join(rootdir, 'config.yaml'), 'r') as file_descriptor:
    config = yaml.load(file_descriptor, Loader=yaml.FullLoader)


from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle

########################################################################################################################
##########################################  TRAIN, VALIDATION, TEST ####################################################
########################################################################################################################



"""Función que recibe como parámetro un diccionario en el que se encuentra la información de las instancias del problema
junto con su clase, y se realiza una separación de manera estratificada en conjuntos de datos de train, validación y test"""
def create_train_validation_test(path_instances, percent_validation, percent_test, path_output):

    #PRECONDICION

    #La precondición es que los porcentajes de validación y test deben de ser un valor entre [0, 1). La suma de ambos porcentajes
    #no puede ser igual a 1, ya que sino no habria instancias en el conjunto de train
    assert 0.0 <= percent_validation < 1.0 and 0.0 <= percent_test < 1.0 and (percent_validation + percent_test) != 1.0

    #LECTURA DE LAS INSTANCIAS
    Path_Instances = Path(path_instances)

    with Path_Instances.open('rb') as file_descriptor:
        data = pickle.load(file_descriptor)

    #Almaceno en una lista el nombre de los ficheros que se encuentran en la carpeta de las instancias
    X = list(data.keys())
    y = list(data.values())

    percent_validation_and_test = percent_validation+percent_test

    #Se obtiene el conjunto de entrenamiento
    X_train, X_validation_and_test, y_train, y_validation_and_test = train_test_split(X, y, test_size=percent_validation_and_test, stratify=y)

    #Se obtienen los conjuntos de validación y de test
    X_validation, X_test, y_validation, y_test = train_test_split(X_validation_and_test, y_validation_and_test, test_size=percent_test/percent_validation_and_test, stratify=y_validation_and_test)

    #Post-condición
    #La suma de elementos de train, validación y test debe de ser igual al número de elementos de la lista inicial de ficheros
    assert len(X_test) + len(X_validation) + len(X_train) == len(X)

    #Escritura de las tres listas en un fichero .txt (para que sea visible por el usuario)

    Path_Output = Path(path_output)

    #Se crea el directorio de salida para almacenar los ficheros en caso de que no exista
    Path_Output.mkdir(parents=True, exist_ok=True)

    with (Path_Output / 'test.txt').open('w') as file_descriptor:
        file_descriptor.writelines("%s\n" % place for place in X_test)

    with (Path_Output / 'validation.txt').open('w') as file_descriptor:
        file_descriptor.writelines("%s\n" % place for place in X_validation)

    with (Path_Output / 'train.txt').open('w') as file_descriptor:
        file_descriptor.writelines("%s\n" % place for place in X_train)

create_train_validation_test(
    path_instances=config['create_train_validation_test']['path_instances'],
    percent_validation=config['create_train_validation_test']['percent_validation'],
    percent_test=config['create_train_validation_test']['percent_test'],
    path_output=config['create_train_validation_test']['path_output']
)
