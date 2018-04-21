from keras import backend as K
from keras.layers import Add, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, Dropout,\
    Flatten, Input, Lambda, MaxPooling2D, Reshape, UpSampling2D
from keras.models import Model
from keras.utils import multi_gpu_model
from tf.contrib.losses.metric_learning import triplet_semihard_loss

def get_model(embedding_size=128, triplet_loss_margin=0.2):
    input_img = Input(shape=(64, 512, 1))

    x = Conv2D(32, (7, 7), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(64, (7, 7), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    #  32, 256, 64
    x = BatchNormalization()(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    # 16, 128, 192
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(320, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    # 8, 64, 320
    x = BatchNormalization()(x)
    x = Conv2D(480, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    # 4, 32, 512
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    # 2, 16, 512
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(embedding_size, activation='relu')(x)

    embedding = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)

    triplet_loss = lambda labels, embeddings: triplet_semihard_loss(labels, embeddings)

    deep_iris_net = Model(inputs=input_img, outputs=embedding)
    deep_iris_net = multi_gpu_model(deep_iris_net, gpus=4)
    deep_iris_net.compile(optimizer='adam', loss=triplet_loss)

    return deep_iris_net
