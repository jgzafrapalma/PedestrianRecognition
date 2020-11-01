
from FuncionesAuxiliares import extract_pedestriansFrames_datasets_PIE




#Función que permite crear instancias que van a ser utilizadas para la evaluación del modelo final
def create_instances_PIE_CrossingDetection(input_path_data, input_path_frames, output_path_cuts, output_path_instances, n_frames, optical_flow=False):

    if not input_path_frames.exists():

        extract_pedestriansFrames_datasets_PIE(input_path_data, input_path_dataset, output_path_frames, rate, shape=())


    logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

    # Se abre el fichero de datos
    with open(input_path_data, 'rb') as file_descriptor:
        data = pickle.load(file_descriptor)

    # Ruta donde se encuentran los frames de cada peaton por video
    Path_Frames = Path(input_path_frames)

    for set_video in Path_Frames.iterdir():

        logging.info("Accediendo al directorio %s" % set_video)

        for video in set_video.iterdir():

            logging.info("Creando instancias del video %s" % video)

            for ped in video.iterdir():

                Path_Ped = Path(join(str(ped), '%04d.jpg'))

                #Se obtiene la etiqueta de si el peatón esta cruzando o no
                crossing = data[set_video.stem][video.stem]['ped_annotations'][ped.stem]['attributes']['crossing']

                #Solamente se va a calcular para aquellos peatones que estan cruzando o no
                if crossing != -1:

                    if optical_flow:

                        Path_Instances = Path(join(output_path_instances, 'Crossing-Detection', str(n_frames) + '_frames', 'OpticalFlow'))

                        Path_Cuts = Path(join(output_path_cuts, 'Crossing-Detection', str(n_frames) + '_frames', 'OpticalFlow', ped.stem))

                        Path_Instances.mkdir(parents=True, exist_ok=True)

                        Path_Cuts.mkdir(parents=True, exist_ok=True)

                        # Comprobar si ya existe la instancia
                        Path_Instance = Path_Instances / (ped.stem + '.pkl')

                        if not Path_Instance.exists():

                            logging.info("Creando instancia %s" % Path_Instance.stem)

                            output_frames = extractFramesOpticalFlow(Path_Ped, n_frames)

                            # Se crea la instancia
                            instance = {'frames': output_frames, 'crossing': crossing}

                            with Path_Instance.open('wb') as file_descriptor:
                                pickle.dump(instance, file_descriptor)

                            for index, frame in enumerate(output_frames):
                                cv2.imwrite(str(Path_Cuts / ('%01d' % (index) + '.jpg')), frame)

                            logging.info("Instancia %s creada con exito" % Path_Instance.stem)

                        else:

                            logging.info("La instancia %s no se ha creado porque ya existe" % Path_Instance.stem)

                    else:

                        Path_Instances = Path(join(output_path_instances, 'Crossing-Detection', str(n_frames) + '_frames', 'Distributed'))

                        Path_Cuts = Path(join(output_path_cuts, 'Crossing-Detection', str(n_frames) + '_frames', 'Distributed', ped.stem))

                        Path_Instances.mkdir(parents=True, exist_ok=True)

                        Path_Cuts.mkdir(parents=True, exist_ok=True)

                        # Comprobar si ya existe la instancia
                        Path_Instance = Path_Instances / (ped.stem + '.pkl')

                        if not Path_Instance.exists():

                            output_frames = extractFrames(Path_Ped, n_frames)

                            # Se crea la instancia
                            instance = {'frames': output_frames, 'crossing': crossing}

                            with Path_Instance.open('wb') as file_descriptor:
                                pickle.dump(instance, file_descriptor)

                            for index, frame in enumerate(output_frames):
                                cv2.imwrite(str(Path_Cuts / ('%01d' % (index) + '.jpg')), frame)

                            logging.info("Instancia %s creada con exito" % Path_Instance.stem)

                        else:

                            logging.info("La instancia %s no se ha creado porque ya existe" % Path_Instance.stem)

create_instances_PIE_CrossingDetection()