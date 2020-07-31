import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45

import argparse

import numpy as np
import random
from tensorflow.compat.v1 import set_random_seed


#COMMAND LINE ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dim', nargs=2, default=(128, 128), type=int, dest='dim', help='dimensionality of input data')

parser.add_argument('-i', '--instances', default='/media/jorge/DATOS/TFG/datasets/instances', type=str, dest='path_instances', help='path of the instances')

parser.add_argument('-b', '--batch', default=32, type=int, dest='batch_size', help='batch size')

parser.add_argument('-e', '--epochs', default=100, type=int, dest='epochs', help='number of epochs')

parser.add_argument('-g', '--logs', default='/media/jorge/DATOS/TFG/logs', type=str, dest='log_dir', help='path to save the logs')

parser.add_argument('-l', '--learning', default=0.001, type=float, dest='learning_rate', help='learning rate value')

parser.add_argument('-o', '--drop', default=0.5, type=float, dest='dropout_rate', help='dropout rate value')

parser.add_argument('-f', '--frames', default=8, type=int, dest='n_frames', help='number of frames')

parser.add_argument('-n', '--normalized', default=False, dest='normalized', help='indicates that the input data is to be normalized', action='store_true')

parser.add_argument('-u', '--suffle', default=False, dest='shuffle', help='indicates if we have a new order of exploration at each pass', action='store_true')

parser.add_argument('-r', '--random', default=False, dest='random', help='Indicate use of random seed', action='store_true')

parser.add_argument('-s', '--seed', type=int, dest='seed', help='seed used for initialization of random generators')


args = parser.parse_args()

print(args.random)

#Si el valor de random es igual a False y no se ha introducido una semilla
if (not args.random and args.seed == None):
    parser.error('When the value of the random argument is False, the seed command must be entered')
#Si el valor de random es true y se ha introducido una semilla
if (args.random and args.seed != None):
    parser.error('When the value of the random argument is True, the seed cannot be inserted')

#Si el valor de random es igual a False y se ha introducido un valor para la semilla se inicializan los generadores
if (not args.random and args.seed != None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed)


from model import create_model

from FuncionesAuxiliares import create_Train_Validation
from FuncionesAuxiliares import save_frames
from DataGenerator import DataGenerator

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



params = {'dim': tuple(args.dim),
          'path_instances': args.path_instances,
          'batch_size': args.batch_size,
          'n_clases': 2,
          'n_channels': 3,
          'n_frames': args.n_frames,
          'normalized': args.normalized,
          'shuffle': args.shuffle}

train_ids_instances, validation_ids_instances = create_Train_Validation(args.path_instances, 0.3)


train_generator = DataGenerator(train_ids_instances, **params)

validation_generator = DataGenerator(validation_ids_instances, **params)

model = create_model((args.n_frames, args.dim[0], args.dim[1], 3), args.dropout_rate, args.learning_rate)

#CALLBACKS

tensorboard = TensorBoard(log_dir=args.log_dir, histogram_freq=1, write_images=True)

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

keras_callbacks = [tensorboard, earlystopping, reducelronplateau]

history = model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=args.epochs, callbacks=keras_callbacks)