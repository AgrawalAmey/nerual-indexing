import datetime
import pickle

from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.svm import SVC

from data_loader import DataLoader, LabelPreservingGenerator
from model import get_model

embedding_size = 128

data_loader = DataLoader()


train_data_generator_wrapper = LabelPreservingGenerator(data_loader.inputs(
    K.get_session(), '../../../data/nd-iris-train-*.tfrecords', 100, 1))

train_data_generator = train_data_generator_wrapper.get_generator()

val_data_generator_wrapper = LabelPreservingGenerator(data_loader.inputs(
    K.get_session(), '../../../data/nd-iris-val-*.tfrecords', 100, 1))

val_data_generator = val_data_generator_wrapper.get_generator()

deep_iris_net = get_model(embedding_size)[0]

# Load weights
deep_iris_net.load_weights(
    '../../checkpoints/deep_iris_net_triplet/1524379063.1205156.49-0.92.hdf5')

print("Getting embeddings for training set.")
train_set_embeddings = deep_iris_net.predict_generator(generator=train_data_generator,
                                                       steps=100,  # batch size 100
                                                       workers=0,
                                                       use_multiprocessing=True)

train_labels = train_data_generator_wrapper.get_labels().reshape(-1)

print("Getting embeddings for validation set.")
val_set_embeddings = deep_iris_net.predict_generator(generator=val_data_generator,
                                                     steps=10,  # batch size 100
                                                     workers=0,
                                                     use_multiprocessing=True)

val_labels = val_data_generator_wrapper.get_labels().reshape(-1)


print('Storing embeddings...')

train_file_name = "../../embeddings/train.p"
val_file_name = "../../embeddings/val.p"

with open(train_file_name, 'wb') as f:
    pickle.dump([train_set_embeddings, train_labels], f)

with open(val_file_name, 'wb') as f:
    pickle.dump([val_set_embeddings, val_labels], f)
