from keras.layers import Add, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, Dropout,\
    Flatten, Input, Lambda, MaxPooling2D, Reshape, UpSampling2D
from keras.models import Model
from keras.utils import multi_gpu_model

def get_model():
    input_img = Input(shape=(64, 512, 1))

    x = Conv2D(32, (7, 7), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(320, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)
    x = Conv2D(480, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)
    x = Flatten(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1352, activation='softmax')(x)

    deep_iris_net = Model(inputs=input_img, outputs=predictions)
    deep_iris_net = multi_gpu_model(deep_iris_net, gpus=4)
    deep_iris_net.compile(optimizer='adam', loss='categorical_crossentropy')

    return deep_iris_net
