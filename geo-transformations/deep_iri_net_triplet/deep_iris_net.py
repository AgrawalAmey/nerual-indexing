import time

from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

from data_loader import DataLoader
from model import get_model

embedding_size = 128

data_loader = DataLoader(embedding_size)

get_train_generator = lambda: data_loader.inputs(K.get_session(), '../../../data/nd-iris-train-*.tfrecords', 100, 50)
get_val_generator = lambda: data_loader.inputs(K.get_session(), '../../../data/nd-iris-val-*.tfrecords', 100, 50)

deep_iris_net = get_model(embedding_size)

deep_iris_net.fit_generator(generator=get_train_generator(),
                epochs=50,
                steps_per_epoch=1024, # batch size 100
                validation_data=get_val_generator(),
                validation_steps=120,
                workers = 0,
                use_multiprocessing=True,
                callbacks=[TensorBoard(log_dir='../../logs/deep_iris_net_triplet/' + str(time.time())),
                           ModelCheckpoint("../../checkpoints/deep_iris_net_triplet/" + str(time.time()) + ".{epoch:02d}-{val_loss:.2f}.hdf5",
                                           save_weights_only=True)])
