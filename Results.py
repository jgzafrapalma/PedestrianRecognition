import DataGenerators

from FuncionesAuxiliares import read_instance_file_txt




"""Inicialización del DataGenerator, en el constructor se inicializa el orden en el que se van a devolver las instancias del problema."""
validation_generator = DataGenerators.DataGeneratorFINALCrossingDetection(validation_ids_instances, **params)

"""Se obtiene los identificadores de las intancias y su etiqueta en el orden en el que son insertadas en el modelo final"""
id_instances_validation, y_validation = validation_generator.get_ID_instances_and_real_labels()

y_predictions = model.predict(validation_generator)

with open('predictions.txt', 'w') as filehandle:
    filehandle.write(str(y_predictions))