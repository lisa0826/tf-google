import tensorflow as tf 

#tf.constant代表常量
a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')
result = a + b
#Session会话用于计算结果
sess = tf.Session()
print(sess.run(result))