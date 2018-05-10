from keras import backend as K
from keras.layers import Add, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, Dropout,\
    Flatten, Input, Lambda, MaxPooling2D, Reshape, UpSampling2D
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf


def get_model():
    input_img = Input(shape=(64, 512, 1))

    x = Conv2D(32, (7, 7), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(64, (7, 7), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    #  32, 256, 64
    x1 = BatchNormalization()(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x1)
    x = BatchNormalization()(x)
    x = Conv2D(192, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # 16, 128, 192
    x2 = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    x = BatchNormalization()(x)
    x = Conv2D(320, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # 8, 64, 320
    x3 = BatchNormalization()(x)
    x = Conv2D(480, (3, 3), activation='relu', padding='same')(x3)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    encoding = MaxPooling2D((2, 2))(x)

    # Decoder begins
    # 4, 32, 512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(encoding)
    x = BatchNormalization()(x)
    x = Conv2D(480, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    #  8, 64, 480
    x = Conv2D(320, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # 16, 128, 256
    x = Conv2D(192, (5, 5), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # 32, 256, 128
    x = Conv2D(64, (7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # 64, 512, 32
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    reconstruction = Conv2D(1, (3, 3), padding='same')(x)

    deep_iris_net = Model(inputs=input_img, outputs=reconstruction)
    deep_iris_net = multi_gpu_model(deep_iris_net, gpus=4)
    deep_iris_net.compile(optimizer='adam', loss='mse')

    return deep_iris_net, input_img, encoding, reconstruction
