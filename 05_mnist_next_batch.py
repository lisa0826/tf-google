# -*- coding:utf-8 -*
from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集，如果指定地址MNIST_data下没有已经下载好的数据，那么tensorflow会自动从表5-1给出的网址下载数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

batch_size = 100
xs,ys = mnist.train.next_batch(batch_size)

# 从train的集合中选取batch_size个训练数据
print "X shape:",xs.shape
# 输出X shape:（100, 784）
print "Y shape:",ys.shape
# 输出Y shape:(100, 10)