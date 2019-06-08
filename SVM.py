import numpy as np
import os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def data_process(path_train):
    trainset = np.loadtxt(path_train, dtype=str)
    train_shape = np.shape(trainset)
    for i in range(train_shape[0]):
        for j in range(train_shape[1]):
            trainset[i][j] = int(trainset[i][j].split(':')[-1])
    print(trainset[0])
    print(train_shape)
    np.save(root + 'test_set.npy', trainset)
    # np.savetxt(root + 'train_set.txt', trainset)


print('======> preparing dataset.')
root = './data/letter/'
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

select_best_dim = True
select_best_C = False
select_kernel = False
best_dim = 100
best_C = 1
best_acc = 0



if select_best_dim:
	dimRange = [50, 100, 150, 200, 250, 300, 500]
	acc_list = []
	for dimRed in dimRange:
		dimRed = 16
		print('======> begin dim reduction.')
		pca = PCA(n_components=dimRed)
		X_train = pca.fit_transform(data_train)
		X_test = pca.transform(data_test)
		print('trainset: ',X_train.shape)
		print('testset: ', X_test.shape)
		print('dim reduced from {} to {}'.format(data_train[0].shape, dimRed))

		print('======> begin fitting.')
		clf = SVC(kernel='linear', C=10)
		clf.fit(X_train, label_train)

		print('======> begin testing.')
		acc = clf.score(X_test, label_test)
		print(dimRed, acc)
		if best_acc < acc:
			best_acc = acc
			best_dim = dimRed
		acc_list.append(acc)
		plt.figure()
		x = [dimRange[i] for i in range(len(acc_list))]
		y = acc_list
		plt.plot(x, y)
		plt.xlabel('dimension')
		plt.ylabel('svm acc')
		# plt.savefig('svm_madelon_dimSelec.png')
		

if select_best_C:
	cRange = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	acc_list = []
	for c in cRange:

		print('======> begin dim reduction.')
		pca = PCA(n_components=16)
		X_train = pca.fit_transform(data_train)
		X_test = pca.transform(data_test)
		print('trainset: ',X_train.shape)
		print('testset: ', X_test.shape)

		print('======> begin fitting.')
		clf = SVC(kernel='linear', C=c)
		clf.fit(X_train, label_train)

		print('======> begin testing.')
		acc = clf.score(X_test, label_test)
		print(c, acc)
		acc_list.append(acc)
		plt.figure()
		x = [cRange[i] for i in range(len(acc_list))]
		y = acc_list
		plt.plot(x, y,color='green', alpha=0.5)
		plt.xlabel('value of C')
		plt.ylabel('svm acc')
		plt.title('param C selection of SVM')
		plt.savefig('svm_madelon_cSelec.png')

if select_kernel:
	kernel_list = [ 'linear', 'poly', 'rbf', 'sigmoid']
	acc_list = []
	for kernel in kernel_list:

		print('======> begin dim reduction.')
		pca = PCA(n_components=16)
		X_train = pca.fit_transform(data_train)
		X_test = pca.transform(data_test)
		print('trainset: ',X_train.shape)
		print('testset: ', X_test.shape)

		print('======> begin fitting.')
		clf = SVC(kernel=kernel, C=10)
		clf.fit(X_train, label_train)

		print('======> begin testing.')
		acc = clf.score(X_test, label_test)
		print(kernel, acc)
		acc_list.append(acc)
		plt.figure()
		x = [kernel_list[i] for i in range(len(acc_list))]
		y = acc_list
		plt.bar(x, y, color='red', alpha=0.5)
		plt.xlabel('kernel')
		plt.ylabel('svm acc')
		plt.title('kernel selected of SVM')
		plt.savefig('svm_madelon_kernelSelec.png')