#神经网络的前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#tf.random_normal([2,3],stddev=2)会产生一个2*3的矩阵，矩阵中的元素是均值为0，标准差为2的随机数。
weights = tf.Variable(tf.random_normal([2,3],stddev=2))

#随机数生成函数
#tf.random_normal 正态分布 平均值、标准差、取值类型
#tf.truncated_normal 正态分布，但如果随机出来的值偏离平均值超过2个标准差，那么这个数将会被重新随机 平均值、标准差、取值类型
#tf.random_uniform 均匀分布 最小、最大取值，取值类型
#tf.random_gamma Gamma分布 形状参数alpha、尺度参数beta、取值类型

#常数生成函数
#tf.zeros 产生全0的数组 tf.zeros([2,3],int32)
#tf.ones 产生全1的数组 tf.ones([2,3],int32)
#tf.fill 产生一个全部为给定数字的数组 tf.fill([2,3],9)
#tf.constant 产生一个给定值的常量 tf.constant([1,2,3])

#在神经网络中，偏置项（bias）通常会使用常数来设置初始值
biases = tf.Variable(tf.zeros([3]))

w2 = tf. V ariable (weights. initialized_value ())
w3 = tf.Variable(weights.initialized_value() * 2.0)