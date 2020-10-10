import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45

from FuncionesAuxiliares import extract_pedestriansFrames_datasets_PIE, create_train_validation_test, extract_pedestriansFrames_datasets_JAAD, create_instances_JAAD, create_instances_PIE_OrderPrediction


"""extract_pedestriansFrames_datasets_JAAD(input_path_data='/media/jorge/DATOS/TFG/data/jaad_database.pkl', input_path_dataset='/media/jorge/DATOS/TFG/datasets/JAAD_clips',
                                        output_pathFrames='/media/jorge/DATOS/TFG/frames/JAAD_dataset', rate= 0.10, shape=(128, 128))"""

"""create_instances_JAAD(input_path_data='/media/jorge/DATOS/TFG/data/jaad_database.pkl', input_path_frames='/media/jorge/DATOS/TFG/frames/JAAD_dataset', output_path_cuts='/media/jorge/DATOS/TFG/cuts/JAAD_dataset',
output_path_instances="/media/jorge/DATOS/TFG/instances/JAAD_dataset", n_frames=16)"""


"""extract_pedestriansFrames_datasets_PIE(input_path_data='/media/jorge/DATOS/TFG/data/pie_database.pkl', input_path_dataset='/media/jorge/DATOS/TFG/datasets/PIE_clips',
                                        output_path_frames='/media/jorge/DATOS/TFG/frames/PIE_dataset', rate= 0.10, shape=(128, 128))"""


"""create_instances_PIE_OrderPrediction(input_path_frames='/media/jorge/DATOS/TFG/frames/PIE_dataset', output_path_cuts='/media/jorge/DATOS/TFG/cuts/PIE_dataset',
output_path_instances="/media/jorge/DATOS/TFG/instances/PIE_dataset", optical_flow=False)"""

"""create_train_validation_test(path_instances='/media/jorge/DATOS/TFG/data/instances_pie_database.pkl', percent_validation=0.15, percent_test=0.15,
                             path_output='/media/jorge/DATOS/TFG/ids_instances/PIE_dataset')"""

#pedestrian = np.load('/media/jorge/DATOS/TFG/datasets/instances/video_0001_0_1_2b.npy')

#ped = extract_Frames_Matriz('/media/jorge/DATOS/TFG/datasets/instances', 'video_0001_0_1_2b.npy', 8)


#create_train_validation_test('/pub/experiments/jzafra/instances/PIE_dataset', 0.3, 0.0, '/pub/experiments/jzafra/ids_instances/PIE_dataset')