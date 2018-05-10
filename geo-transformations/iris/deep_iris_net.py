
# coding: utf-8

# In[1]:


import itertools
import os
import sys
import time

from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Add, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, Input, Lambda, MaxPooling2D, Reshape, UpSampling2D
from keras.models import Model
from keras.utils import multi_gpu_model
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from data_loader import inputs
from model_din import get_model
from visdom_callback import PlotVisdom

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# debugging
# K.set_session(tf_debug.TensorBoardDebugWrapperSession(K.get_session(),
#                                                       'localhost:6064',

#show_samples_from_tfr('../../data/nd-iris-train-*.tfrecords')

autoencoder = get_model()[0]

get_train_generator = lambda: inputs(K.get_session(), '../../../data/nd-iris-t*.tfrecords', 400, 25)
get_val_generator = lambda: inputs(K.get_session(), '../../../data/nd-iris-val-*.tfrecords', 400, 25)


autoencoder.fit_generator(generator=get_train_generator(),
                epochs=25,
                steps_per_epoch=256, # batch size 400
                validation_data=get_val_generator(),
                validation_steps=30,
                workers = 0,
                use_multiprocessing=True,
                          callbacks=[TensorBoard(log_dir='../../logs/iris_ae_din' + str(time.time())),
			               ModelCheckpoint("../../checkpoints/iris_ae_din" + str(time.time()) + ".{epoch:02d}-{val_loss:.2f}.hdf5",
                                           save_weights_only=True),
                           PlotVisdom(autoencoder, get_train_generator, get_val_generator)])
