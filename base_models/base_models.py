from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Conv2D, MaxPooling2D, MaxPooling3D, BatchNormalization



class CaffeNet(Model):

    def __init__(self, the_input_shape):
        super(CaffeNet, self).__init__()

        # 1th Convolutional Layers
        self.Conv2D_1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', data_format='channels_last',
                     activation='relu', input_shape=the_input_shape, name='Conv2D_1_CaffeNet')
        self.MaxPooling2D_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_1_CaffeNet')
        self.BatchNormalization_1 = BatchNormalization(name='BatchNormalization_1_CaffeNet')

        # 2th Convolutional Layers
        self.Conv2D_2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last',
                     activation='relu', name='Conv2D_2_CaffeNet')
        self.MaxPooling2D_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_2_CaffeNet')
        self.BatchNormalization_2 = BatchNormalization(name='BatchNormalization_2_CaffeNet')

        # 3th Convolutional Layers
        self.Conv2D_3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_3_CaffeNet')

        # 4th Convolutional Layers
        self.Conv2D_4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_4_CaffeNet')

        # 5th Convolutional Layers
        self.Conv2D_5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                    activation='relu', name='Conv2D_5_CaffeNet')
        self.MaxPooling2D_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last', name='MaxPooling2D_3_CaffeNet')

    def call(self, inputs, training=True):
        # 1th Convolutional Layers
        x = self.Conv2D_1(inputs)
        x = self.MaxPooling2D_1(x)
        x = self.BatchNormalization(x, training)

        # 2th Convolutional Layers
        x = self.Conv2D_2(x)
        x = self.MaxPooling2D_2(x)
        x = self.BatchNormalization_2(x, training)

        # 3th Convolutional Layers
        x = self.Conv2D_3(x)

        # 4th Convolutional Layers
        x = self.Conv2D_4(x)

        # 5th Convolutional Layers
        x = self.Conv2D_5(x)
        outputs = self.MaxPooling2D_3(x)

        return outputs



class CONV3D(Model):

    def __init__(self, the_input_shape):
        super(CONV3D, self).__init__()

        self.the_input_shape = the_input_shape

        # 1th Convolutional Layer
        self.Conv3D_1 = Conv3D(16, (3, 5, 5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                            activation='relu', input_shape=the_input_shape, name='Conv3D_1_CONV3D')

        # 2th Convolutional Layer
        self.Conv3D_2 = Conv3D(24, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                            activation='relu', name='Conv3D_2_CONV3D')

        # 3th Convolutional Layer
        self.Conv3D_3 = Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                            activation='relu', name='Conv3D_3_CONV3D')

        # 4th Convolutional Layer
        self.Conv3D_4 = Conv3D(12, (1, 6, 6), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                            activation='relu', name='Conv3D_4_CONV3D')

    def call(self, inputs, training=True):
        # 1th Convolutional Layer
        x = self.Conv3D_1(inputs)

        # 2th Convolutional Layer
        x = self.Conv3D_2(x)

        # 3th Convolutional Layer
        x = self.Conv3D_3(x)

        # 4th Convolutional Layer
        outputs = self.Conv3D_4(x)

        return outputs



class C3D(Model):

    def __init__(self, the_input_shape):

        super(C3D, self).__init__()

        self.the_input_shape = the_input_shape

        # 1th Convolutional Layers
        self.Conv1a = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last',
                             activation='relu', input_shape=the_input_shape, name='Conv1a_C3D')
        #self.BatchNormalization_1 = BatchNormalization(name='BatchNormalization_1_C3D')
        self.Pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', data_format='channels_last', name='Pool1_C3D')

        # 2th Convolutional Layers
        self.Conv2a = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last',
                             activation='relu', name='Conv2a_C3D')
        #self.BatchNormalization_2 = BatchNormalization(name='BatchNormalization_2_C3D')
        self.Pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_last', name='Pool2_C3D')

        # 3th Convolutional Layers
        self.Conv3a = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last',
                             activation='relu', name='Conv3a_C3D')
        #self.BatchNormalization_3 = BatchNormalization(name='BatchNormalization_3_C3D')
        self.Conv3b = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last',
                             activation='relu', name='Conv3b_C3D')
        #self.BatchNormalization_4 = BatchNormalization(name='BatchNormalization_4_C3D')
        self.Pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_last', name='Pool3_C3D')

        # 4th Convolutional Layers
        self.Conv4a = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last',
                             activation='relu', name='Conv4a_C3D')
        #self.BatchNormalization_5 = BatchNormalization(name='BatchNormalization_5_C3D')
        self.Conv4b = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last',
                             activation='relu', name='Conv4b_C3D')
        #self.BatchNormalization_6 = BatchNormalization(name='BatchNormalization_6_C3D')
        self.Pool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_last', name='Pool4_C3D')

        # 5th Convolutional Layers
        self.Conv5a = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last',
                             activation='relu', name='Conv5a_C3D')
        #self.BatchNormalization_7 = BatchNormalization(name='BatchNormalization_7_C3D')
        self.Conv5b = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last',
                             activation='relu', name='Conv5b_C3D')
        #self.BatchNormalization_8 = BatchNormalization(name='BatchNormalization_8_C3D')
        self.Pool5 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_last', name='Pool5_C3D')


    def call(self, inputs, training=True):
        # 1th Convolutional Layer
        x = self.Conv1a(inputs)

        #x = self.BatchNormalization_1(x, training)

        x = self.Pool1(x)

        # 2th Convolutional Layer
        x = self.Conv2a(x)

        #x = self.BatchNormalization_2(x, training)

        x = self.Pool2(x)

        # 3th Convolutional Layer
        x = self.Conv3a(x)

        #x = self.BatchNormalization_3(x, training)

        x = self.Conv3b(x)

        #x = self.BatchNormalization_4(x, training)

        x = self.Pool3(x)

        # 4th Convolutional Layer
        x = self.Conv4a(x)

        #x = self.BatchNormalization_5(x, training)

        x = self.Conv4b(x)

        #x = self.BatchNormalization_6(x, training)

        x = self.Pool4(x)

        # 5th Convolutional Layer
        x = self.Conv5a(x)

        #x = self.BatchNormalization_7(x, training)

        x = self.Conv5b(x)

        #x = self.BatchNormalization_8(x, training)

        outputs = self.Pool5(x)

        return outputs
