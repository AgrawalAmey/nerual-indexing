import itertools
import os
import sys
import time

from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

from data_loader import DataLoader
from model import get_model


data_loader = DataLoader(1352)

get_train_generator = lambda: data_loader.inputs(K.get_session(), '../../../data/nd-iris-train-*.tfrecords', 400, 25)
get_val_generator = lambda: data_loader.inputs(K.get_session(), '../../../data/nd-iris-val-*.tfrecords', 400, 25)

deep_iris_net = get_model()

deep_iris_net.fit_generator(generator=get_train_generator(),
                epochs=25,
                steps_per_epoch=256, # batch size 400
                validation_data=get_val_generator(),
                validation_steps=30,
                workers = 0,
                use_multiprocessing=True,
                callbacks=[TensorBoard(log_dir='../../logs/deep_iris_net/' + str(time.time())),
                           ModelCheckpoint("../../checkpoints/deep_iris_net/" + str(time.time()) + ".{epoch:02d}-{val_loss:.2f}.hdf5",
                                           save_weights_only=True)])
