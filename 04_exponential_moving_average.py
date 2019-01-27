import tensorflow as tf 

#定义一个变量用于计算滑动平均，这个变量的初始值为0，注意这里手动指定了变量的类型为tf.float32,因为所有需要计算滑动平均的变量必需是实数型
v1  = tf.Variable(0,dtype=tf.float32)
#这里step变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)

#定义一个滑动平均的类（class），初始化时给定了衰减率（0.99）和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99,step)
