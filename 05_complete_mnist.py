import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#MNIST数据集相关的常数
INPUT_NODE = 784   #输入层的节点数，对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE = 10   #输出层的节点数，这个等于类别的数目，因为在MNIST数据集中需要区分的是0~9这10个数字，所以这里输出层的节点数为10

#配置神经网络的参数
LAYER1_NODE = 500  #隐藏层节点数，这里使用只有一个隐藏层的网络结构作为样例，这个隐藏层有500个节点
BATCH_SIZE = 100   #一个训练batch中的训练数据个数，数字越小时，训练过程越接近随机梯度下降：数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.8   #基础的学习率
LEARNING_RATE_DECAY = 0.99   #学习率的衰减率
REGULARIZATION_RATE = 0.0001   #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000   #训练轮数
MOVING_AVERAGE_DECAY = 0.99   #滑动平均衰减率

#一个辅助函数，给定神经网络的输入