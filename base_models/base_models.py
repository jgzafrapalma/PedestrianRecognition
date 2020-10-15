from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Conv2D, MaxPooling2D, BatchNormalization



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