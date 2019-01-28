# -*- coding:utf-8 -*
from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集，如果指定地址/path/to/MNIST_data下没有已经下载好的数据，那么tensorflow会自动从表5-1给出的网址下载数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# 打印Training data size:55000
print "Training data size:",mnist.train.num_examples

# 打印Validating data size:5000
print "Validating data size:",mnist.validation.num_examples

# 打印Testing data size:10000
print "Testing data size:",mnist.test.num_examples

# 打印Example training data:
print "Example training data:",mnist.train.images[1]

# 打印Example training data label:
print "Example training data label:",mnist.train.labels[0]