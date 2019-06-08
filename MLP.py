import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

print(__doc__)

print('======> preparing dataset.')
root = './data/cifar-10/'
path_train = root + 'train_set.npy'
trainset = np.load(path_train).astype(np.float)

path_test = root + 'test_set.npy'
testset = np.load(path_test).astype(np.float)

label_train = [trainset[i][0] for i in range(trainset.shape[0])]
data_train = [trainset[i][1:] for i in range(trainset.shape[0])]
label_train = np.array(label_train)
data_train = np.array(data_train)

label_test = [testset[i][0] for i in range(testset.shape[0])]
data_test = [testset[i][1:] for i in range(testset.shape[0])]
label_test = np.array(label_test)
data_test = np.array(data_test)
print('dataset available.')

print('trainset: ',data_train.shape)
print('testset: ', data_test.shape)

# rescale the data, use the traditional train/test split
X_train, X_test = data_train, data_test
y_train, y_test = label_train, label_test

# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                    solver='adam', verbose=10, tol=0, random_state=1,
                    learning_rate_init=.01)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

