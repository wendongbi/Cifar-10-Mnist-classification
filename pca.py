import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def getShape(M):
	return (len(M), len(M[0]))
print('begin loading data')
train_X = np.genfromtxt(fname='./data_process(float)/data_train.txt',
							dtype=np.float, max_rows=37322-14929, skip_header=0)
test_X = np.genfromtxt(fname='./data_process(float)/data_test.txt',
							dtype=np.float, max_rows=14929, skip_header=0)
print('data load done.')
print('begin loading label')
train_y = np.genfromtxt(fname='./data_process(float)/label_train.txt',
							dtype=np.int, max_rows=37322-14929, skip_header=0)
test_y = np.genfromtxt(fname='./data_process(float)/label_test.txt',
							dtype=np.int, max_rows=14929, skip_header=0)
print('label load done.')

print('data loading done.')
dim_range = [895, 897]
# pca = PCA(n_components='mle', svd_solver='full')
for DimRedu in dim_range:
	pca = PCA(n_components=DimRedu)
	print(np.shape(train_X))
	trainX_new = pca.fit_transform(train_X)
	print(np.shape(trainX_new))
	testX_new = pca.transform(test_X)

	print(getShape(testX_new))

	trainX_name = './data_process(float)/pca/pca_dim_' + str(DimRedu) + '_train_data.txt'
	testX_name = './data_process(float)/pca/pca_dim_' + str(DimRedu) + '_test_data.txt'
	trainy_name = './data_process(float)/pca/pca_dim_' + str(DimRedu) + '_train_label.txt'
	testy_name = './data_process(float)/pca/pca_dim_' + str(DimRedu) + '_test_label.txt'

	np.savetxt(trainX_name, trainX_new)
	np.savetxt(testX_name, testX_new)
	np.savetxt(trainy_name, train_y)
	np.savetxt(testy_name, test_y)
	print('pca(dim=' + str(DimRedu) + ') finished.\n')
print('All dim_reduction have done.')