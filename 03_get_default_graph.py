# -*- coding: utf-8 -*
import tensorflow as tf 

#tf.constant代表常量
a = tf.constant([1.0,2.0],name='a')

#get_default_graph函数可以获取当前默认的计算图，在tensorflow程序中，系统会自动维护一个默认的计算图
print(a.graph is tf.get_default_graph())
#true