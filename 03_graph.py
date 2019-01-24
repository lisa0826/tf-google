# -*- coding: utf-8 -*
import tensorflow as tf 

g1 = tf.Graph()
with g1.as_default():
	#在计算图g1中定义变量“v”，并设置初始值为0.
	a = tf.get_variable('a',initializer = tf.zeros_initializer()(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
	#在计算图g1中定义变量“v”，并设置初始值为1.
	b = tf.get_variable('b',initializer = tf.ones_initializer()(shape=[1]))

#在计算图g1中读取变量“v”的取值。
with tf.Session(graph=g1) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope("", reuse = True):
		#在计算图g1中，变量“v“的取值应该为0，所以下面这行会输出[0.]
		print(sess.run(tf.get_variable("a")))

#在计算图g2中读取变量“v”的取值。
with tf.Session(graph=g2) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope("", reuse = True):
		#在计算图g1中，变量“v“的取值应该为1，所以下面这行会输出[1.]
		print(sess.run(tf.get_variable("b")))


##报错记录
#1、错误“_init_() got an unexpected keyword argument ‘shape’”
#如果按照书上的例子来，因为这本书使用tensorflow是0.9.0版本，而在最新的tensorflow中有很多改动，文章最后会附上这些改动以供参考查看。这里的错误是因为新版tf.zeros_initializer和tf.ones_initializer后面需要加括号，将v = tf.get_variable(“v”,initializer=tf.zeros_initializer(shape=[1]))改为v = tf.get_variable(“v”,initializer=tf.zeros_initializer( )(shape=[1]))就可以了，下面的一样。

#2、ValueError: Variable foo/bar already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at: 
#如上报错，一般是同一个空间，变量重命名了 