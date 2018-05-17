from keras import backend as K
from keras.layers import Add, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, Dropout,\
    Flatten, Input, Lambda, LSTM, MaxPooling2D, Reshape, TimeDistributed, UpSampling2D
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf


def get_encoder_cnn(input_img):
    # 8, 64, 64, 1
    x = TimeDistributed(Conv2D(
        16, (3, 3), activation='relu', padding='same'))(input_img)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    # 8, 32, 32, 32
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(64, (5, 5), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(64, (5, 5), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    # 8, 16, 16, 64
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    # 8, 8, 8, 16
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same'))(x)
    cnn_encodings = TimeDistributed(Flatten())(x)

    # 8, 64
    return cnn_encodings


def get_seq_to_seq(cnn_encodings):
    # 8, 64
    x = LSTM(256, return_sequences=True)(cnn_encodings)
    # 8, 256
    embedding = LSTM(512)(x)
    # 512
    x = LSTM(256, return_sequences=True)(embedding)
    # 8, 




    deep_iris_net = Model(inputs=input_img, outputs=reconstruction)
    deep_iris_net = multi_gpu_model(deep_iris_net, gpus=4)
    deep_iris_net.compile(optimizer='adam', loss='mse')


input_img = Input(shape=(8, 64, 64, 1))
    
    # Decoder begins
    # 4, 32, 512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(encoding)
    x = BatchNormalization()(x)
    x = Conv2D(480, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    #  8, 64, 480
    x = BatchNormalization()(x)
    x = Conv2D(320, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # 16, 128, 256
    x = BatchNormalization()(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # 32, 256, 128
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # 64, 512, 32
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    reconstruction = Conv2D(1, (3, 3), padding='same')(x)
