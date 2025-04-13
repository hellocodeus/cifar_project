import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from train import load_cifar10
from model import ThreeLayerNet
from test import visualize_weights,visualize_accuracy,load_model_weights
from hyperparameter import hyperparameter_search
from test import visualize_accuracy

if __name__ == "__main__":
    cifar10_dir = './cifar-10-batches-py'
    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10(cifar10_dir)
    print("已加载数据集！")
    best_net = hyperparameter_search(x_train, y_train, x_val, y_val)
    # best_net = ThreeLayerNet(x_train.shape[1], 600, 10)
    stats = best_net.train(x_train, y_train, x_val, y_val,
                           learning_rate=5e-3, learning_rate_decay=0.99,
                           batch_size=600, reg=5e-4,
                           num_iters=20000, verbose=True)
    test_acc = (best_net.predict(x_test) == y_test).mean()
    print(f'Test accuracy: {test_acc}')
    visualize_accuracy(stats)
    visualize_weights(best_net)
    with open('best_net_weights.pkl', 'wb') as f:
        pickle.dump(best_net.params, f)
    
    # # 如果需要从保存的模型权重加载模型进行测试，请取消下面的注释
    # input_size = x_train.shape[1]
    # hidden_size = 512  # 这里需要和训练时的参数一致
    # output_size = 10
    # loaded_net = load_model_weights('./best_model.pkl', input_size, hidden_size, output_size)
    # if loaded_net:
    #     loaded_test_acc = (loaded_net.predict(x_test) == y_test).mean()
    #     print(f'Test accuracy of loaded model: {loaded_test_acc}')