import numpy as np
import os
from sklearn.svm import SVC



def data_process(root, file_path,sample_size, save_path = 'train_set.npy'):
    f = open(file_path, 'r')
    dataset = np.zeros(sample_size)
    idx_line = 0
    while True:
        lines = f.readline()
        if not lines:
            break
        lines = [i for i in lines.split()]

        for i in range(len(lines)):
            if i == 0:
                dataset[idx_line][0] = float(lines[i])
            else:
                print('===>', lines[i].split(':'))
                print(lines[i])
                dataset[idx_line][ int( lines[i].split(':')[0] ) ] = float(lines[i].split(':')[1])
        idx_line+=1
    print(idx_line)
                
    print(dataset.shape)
    print(type(dataset[0][0]))
    np.save(root + save_path, dataset)


# print('======> preparing dataset.')
# root = './data/madelon/'
# path_train = root + 'train_set.npy'
# trainset = np.load(path_train).astype(np.float)

# path_test = root + 'test_set.npy'
# testset = np.load(path_test).astype(np.float)

# label_train = [trainset[i][0] for i in range(trainset.shape[0])]
# data_train = [trainset[i][1:] for i in range(trainset.shape[0])]
# label_train = np.array(label_train)
# data_train = np.array(data_train)

# label_test = [testset[i][0] for i in range(testset.shape[0])]
# data_test = [testset[i][1:] for i in range(testset.shape[0])]
# label_test = np.array(label_test)
# data_test = np.array(data_test)
# print('dataset available.')

# print(trainset.shape)
# print(trainset[0][0])
# print(type(trainset[0][0]))

# print(testset[0])
# print(testset.shape)
# print(type(testset[0][0]))

if __name__ == "__main__":
    root = './data/cifar-10/'
    # path_train = root + 'train.txt'
    # save_path = 'train_set.npy'
    # data_process(root=root, file_path=path_train, save_path=save_path, sample_size=(50000, 3072+1))

    path_train = root + 'test.txt'
    save_path = 'test_set.npy'
    data_process(root=root, file_path=path_train, save_path=save_path, sample_size=(10000, 3072+1))

    f = np.load(root + 'test_set.npy')
    print(f.shape)
    print(f[10000-1][0], f[2][0])
    # print(f[0])