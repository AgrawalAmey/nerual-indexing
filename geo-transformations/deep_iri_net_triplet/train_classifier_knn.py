import datetime
import pickle

from sklearn.neighbors import KNeighborsClassifier

# Load embeddings
print('Loading embeddings...')

train_file_name = "../../embeddings/train.p"
val_file_name = "../../embeddings/val.p"

X_train, y_train = pickle.load(open(train_file_name, 'rb'))
X_val, y_val = pickle.load(open(val_file_name, 'rb'))

classifier = KNeighborsClassifier(n_neighbors=2)

# Fit the model
print('Fitting the model...')
classifier.fit(X_train, y_train)

# Score
print('Evaluating on train set...')
train_score = classifier.score(X_train, y_train)

print('Evaluating on val set...')
val_score = classifier.score(X_val, y_val)

print("Train score: {}, Val score: {}.".format(
                        train_score, val_score))

print("Saving classifier model...")

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_name = "../../checkpoints/deep_iris_net_triplet_knn/{}_{}_{}.p".format(timestamp,
                                                                            train_score,
                                                                            val_score)
with open(file_name, 'wb') as f:
    pickle.dump(classifier, f)
