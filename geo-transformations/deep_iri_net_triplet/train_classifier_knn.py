import datetime
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load embeddings
print('Loading embeddings...')

train_file_name = "../../embeddings/train.p"

X, y = pickle.load(open(train_file_name, 'rb'))

classifier = KNeighborsClassifier(n_neighbors=5)

print('Splitting data...')
X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42,
                shuffle=True)

# Fit the model
print('Fitting the model...')
classifier.fit(X_train, y_train)

# Score
print('Evaluating...')
test_score = classifier.score(X_test, y_test)

print("Test score: {}.".format(
                        test_score))

print("Saving classifier model...")

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_name = "../../checkpoints/deep_iris_net_triplet_knn/{}_{}.p".format(timestamp,
                                                                            test_score)
with open(file_name, 'wb') as f:
    pickle.dump(classifier, f)
