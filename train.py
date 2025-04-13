import numpy as np  
import os

def load_batch(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']


def load_cifar10(cifar10_dir):
    x_train = []
    y_train = []
    for i in range(1, 6):
        file = os.path.join(cifar10_dir, f'data_batch_{i}')
        data, labels = load_batch(file)
        x_train.extend(data)
        y_train.extend(labels)
    x_train = np.array(x_train).astype(np.float64) / 255.0
    y_train = np.array(y_train)
    file = os.path.join(cifar10_dir, 'test_batch')
    x_test, y_test = load_batch(file)
    x_test = np.array(x_test).astype(np.float64) / 255.0
    y_test = np.array(y_test)
    num_train = int(0.8 * x_train.shape[0])
    x_val = x_train[num_train:]
    y_val = y_train[num_train:]
    x_train = x_train[:num_train]
    y_train = y_train[:num_train]
    return x_train, y_train, x_val, y_val, x_test, y_test
