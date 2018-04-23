import datetime
import pickle

from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.svm import SVC

from data_loader import DataLoader, LabelPreservingGenerator
from model import get_model

embedding_size = 128

data_loader = DataLoader(embedding_size)


train_data_generator_wrapper = LabelPreservingGenerator(data_loader.inputs(
    K.get_session(), '../../../data/nd-iris-train-*.tfrecords', 100, 50))

train_data_generator = train_data_generator_wrapper.get_generator()

val_data_generator_wrapper = LabelPreservingGenerator(data_loader.inputs(
    K.get_session(), '../../../data/nd-iris-val-*.tfrecords', 100, 50))

val_data_generator = val_data_generator_wrapper.get_generator()

deep_iris_net = get_model(embedding_size)

# Load weights
deep_iris_net = deep_iris_net.load_weights(
    '../../checkpoints/deep_iris_net_triplet/1524379063.1205156.49-0.92.hdf5')

print("Getting embeddings for training set.")
train_set_embeddings = deep_iris_net.predict_generator(generator=train_data_generator,
                                                       steps=1024,  # batch size 100
                                                       workers=0,
                                                       use_multiprocessing=True)

train_labels = train_data_generator_wrapper.get_labels().reshape(-1)

print("Getting embeddings for validation set.")
val_set_embeddings = deep_iris_net.predict_generator(generator=val_data_generator,
                                                     steps=1024,  # batch size 100
                                                     workers=0,
                                                     use_multiprocessing=True)

val_labels = val_data_generator_wrapper.get_labels().reshape(-1)


print('Training classifier')

model = SVC(kernel='linear', probability=True)
model.fit(train_set_embeddings, train_labels)

print('Checking performance on train set')
train_score = model.score(train_set_embeddings, train_labels)
print("Training set accuracy: {}".format(train_score))

print('Checking performance on validation set')
val_score = model.score(val_set_embeddings, val_labels)
print("Validation set accuracy: {}".format(val_score))

print("Saving classifier model")

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_name = "../../checkpoints/deep_iris_net_triplet_svc/{}_{}_{}".format(timestamp,
                                                                          train_score,
                                                                          val_score)
with open(file_name, 'wb') as outfile:
    pickle.dump((model, class_names), outfile)

print("Saved classifier model.")
