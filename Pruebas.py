import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45

from FuncionesAuxiliares import extract_pedestrian_dataset_PIE, create_train_validation_test

"""extract_pedestrian_dataset_PIE(input_path_data='/media/jorge/DATOS/TFG/data/pie_database.pkl', input_path_dataset='/media/jorge/DATOS/TFG/datasets/PIE_clips',
                               output_path_instances='/media/jorge/DATOS/TFG/instances/PIE_pedestrians', output_path_frames='/media/jorge/DATOS/TFG/frames/PIE_frames',
                               output_path_cuts='/media/jorge/DATOS/TFG/cuts/PIE_cuts', rate=0.10, n_frames=16, shape=(128, 128))"""

""""extract_pedestrian_dataset_PIE(input_path_data='/pub/experiments/jzafra/data/pie_database.pkl', input_path_dataset='/pub/experiments/jzafra/datasets/PIE_clips',
                               output_path_instances='/pub/experiments/jzafra/instances/PIE_dataset', output_path_frames='/pub/experiments/jzafra/frames/PIE_frames',
                               output_path_cuts='/pub/experiments/jzafra/cuts/PIE_cuts', rate=0.10, n_frames=16, shape=(128, 128))"""




#pedestrian = np.load('/media/jorge/DATOS/TFG/datasets/instances/video_0001_0_1_2b.npy')

#ped = extract_Frames_Matriz('/media/jorge/DATOS/TFG/datasets/instances', 'video_0001_0_1_2b.npy', 8)


create_train_validation_test('/pub/experiments/jzafra/instances/PIE_dataset', 0.3, 0.0, '/pub/experiments/jzafra/ids_instances/PIE_dataset')