import datetime
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

# Load embeddings
print('Loading embeddings...')

train_file_name = "../../embeddings/train.p"
val_file_name = "../../embeddings/val.p"

X_train, y_train = pickle.load(open(train_file_name, 'rb'))
X_val, y_val = pickle.load(open(val_file_name, 'rb'))

print('Sorting datapoints by labels...')

train_argsort = np.argsort(y_train)
X_train. y_train =  X_train[train_argsort], y_train[train_argsort]

val_argsort = np.argsort(y_val)
X_val. y_val = X_val[val_argsort], y_val[val_argsort]

print('Calculating distance matrix...')

D_train = pairwise_distances(X_train, n_jobs=-1)
D_val = pairwise_distances(X_val, n_jobs=-1)

print('Storing the matrix')

plt.figure()
plt.colorbar()
plt.title('Train distance matrix')
plt.imshow(D_train, cmap=plt.cm.Blues)
plt.xticks(np.arange(y_train.shape[0]), y_train, rotation=45)
plt.yticks(np.arange(y_train.shape[0]), y_train)
plt.savefig('./train_dist.png')

plt.figure()
plt.colorbar()
plt.title('Val distance matrix')
plt.imshow(D_val, cmap=plt.cm.Blues)
plt.xticks(np.arange(y_val.shape[0]), y_val, rotation=45)
plt.yticks(np.arange(y_val.shape[0]), y_val)
plt.savefig('./test_dist.png')
