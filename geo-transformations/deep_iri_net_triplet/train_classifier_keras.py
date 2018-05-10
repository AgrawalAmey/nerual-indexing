import time

from keras import backend as K
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model

from data_loader import DataLoader
from model import get_model

embedding_size = 128

data_loader = DataLoader()

def get_train_generator(): return data_loader.inputs(
    K.get_session(), '../../../data/nd-iris-train-*.tfrecords', 100, 50)


def get_val_generator(): return data_loader.inputs(
    K.get_session(), '../../../data/nd-iris-val-*.tfrecords', 100, 50)

deep_iris_net, input_img, embedding = get_model(embedding_size)

# Load weights
deep_iris_net.load_weights(
    '../../checkpoints/deep_iris_net_triplet/1524379063.1205156.49-0.92.hdf5')


print('Training classifier')

# Define model
x = Dense(512, activation='tanh')(embedding)
x = Dense(1352, activation='softmax')(x)

model = Model(inputs=input_img, outputs=embedding)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


model.fit_generator(generator=get_train_generator(),
                    epochs=50,
                    steps_per_epoch=1000,  # batch size 100
                    validation_data=get_val_generator(),
                    validation_steps=100,
                    workers=0,
                    use_multiprocessing=True,
                    callbacks=[TensorBoard(log_dir='../../logs/deep_iris_net_triplet_softmax/' + str(time.time())),
                                ModelCheckpoint("../../checkpoints/deep_iris_net_triplet_softmax/" + str(time.time()) + ".{epoch:02d}-{val_loss:.2f}.hdf5")])

