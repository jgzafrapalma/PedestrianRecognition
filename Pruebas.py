import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45

from FuncionesAuxiliares import extract_pedestrians_datasets
from FuncionesAuxiliares import extract_Frames_Matriz

#extract_pedestrians_datasets('/media/jorge/DATOS/TFG/datasets', './JAAD-JAAD_2.0/data_cache/jaad_database.pkl', 0.05, (128, 128))
extract_pedestrians_datasets(pathVideos='/pub/databases/JAAD/JAAD_clips/', pathInstances='/pub/experiments/jzafra/Datasets/JAAD_Instances'
                             , pathFrames='', pathData='./jaad_database.pkl', rate=0.05, shape=(128, 128), frames=False)

#pedestrian = np.load('/media/jorge/DATOS/TFG/datasets/instances/video_0001_0_1_2b.npy')

#ped = extract_Frames_Matriz('/media/jorge/DATOS/TFG/datasets/instances', 'video_0001_0_1_2b.npy', 8)

