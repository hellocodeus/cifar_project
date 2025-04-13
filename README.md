# cifar_project
DATA620004 神经网络和深度学习

# 三层神经网络的CIFAR图像识别
使用numpy搭建三层神经网络分类器，在数据集cifar-10上进行训练以实现图像分类。
## 训练方法
1. 修改main.py中的cifar10_dir变量，使之指向cifar-10训练数据所存放的文件路径
2. 运行main.py即可进行模型训练和测试，并且训练结束后可以输出Loss和Accuracy的变化曲线。
## 加载best_net_weights.pkl模型权重进行测试
1. 在https://drive.google.com/file/d/1gCIXVgqsgAbIxunwOO5aXgUDQARX-FlJ/view?usp=sharing下载已经训练好的模型权重
2. 在main.py中修改加载模型权重部分的代码，将hidden_size设置成和训练时的参数一致，并且指定load_model_weights中模型权重存放的路径
3. 运行main.py即可进行测试

训练数据集链接：https://www.cs.toronto.edu/~kriz/cifar.html
