import matplotlib.pyplot as plt
import numpy as np

from model import ThreeLayerNet
import pickle

def visualize_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    num_filters = W1.shape[0]
    plt.figure(figsize=(10, 10))
    for i in range(num_filters):
        plt.subplot(int(np.sqrt(num_filters)), int(np.sqrt(num_filters)), i + 1)
        filter_img = (W1[i] - np.min(W1[i])) / (np.max(W1[i]) - np.min(W1[i]))
        plt.imshow(filter_img)
        plt.axis('off')
    plt.show()

def visualize_accuracy(stats):
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    plt.show()

def load_model_weights(model_path, input_size, hidden_size, output_size, activation='relu'):
    net = ThreeLayerNet(input_size, hidden_size, output_size, activation)
    try:
        with open(model_path, 'rb') as f:
            weights = pickle.load(f)
            net.params = weights
        return net
    except FileNotFoundError:
        print(f"Error: 文件 {model_path} 未找到.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
