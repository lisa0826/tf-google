# -*- coding: utf-8 -*
import tensorflow as tf 

w1 = tf.Variable(tf.random_normal([2,3],stddev=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1))

#定义placeholder作为存放输入数据的地方，这里维度也不一定要定义。但如果维度是确定的，那么给出维度可以降低出错的概率
x = tf.placeholder(tf.float32,shape=(3,2),name="input")

#通过3.4.2节描述的前向传播算法获得神经网络的输出
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

#下面一行将报错：InvalidArgumentError:You must feed a value for placeholder tensor 'input_l' with dtype float and shape [1,2]
# print(sess.run(y))

#下面一行将会得到和3.4.2节中一样的输出结果：[[0.03460053]]
print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
# sess.close()