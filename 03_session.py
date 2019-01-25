# 创建一个会话
sess = tf.Session()
# 使用这个创建好的会话来得到关心的运算的结果。比如可以调用sess.run(result),来得到 3.1 节样例中张量result的取值
sess.run(...)
# 关闭会话使得本次运行中使用到的资源可以被释放
sess.close()


# 创建一个会话，并通过python中的上下文管理器来管理这个会话
# 使用这个创建好的会话来得到关心的运算的结果，比如可以调用sess.run(result)，来得到3.1节样例中张量result的取值
sess.run(...)
#关闭会话使得本次运行中使用到的资源可以被释放
sess.close()


# 创建一个会话，并通过 Python 中上下文管理器来管理这个会话
with tf.Session() as sess:
	# 使用创建好的会话来计算关心的结果
	sess.run(...)
# 不需要再调用”Session.close()“函数来关闭会话
# 当上下文退出时会话关闭和资源释放也自动完成了 


#通过设定默认会话计算张量的取值
sess = tf.Session()
with sess.as_default():
	print(result.eval())

#功能同上
print(sess.run(result))
print(result.eval(session=sess))


#交互式环境下直接构建默认会话的函数，使用tf.InteractiveSession这个函数会自动将生成的会话注册为默认会话
sess = tf.InteractiveSession()
print(result.eval())
sess.close()

#通过ConfigProto配置会话的方法：
config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
#allow_soft_placement,这是一个布尔型的参数，当它为True时，在以下任意一个条件成立时，GPU上的运算可以放到CPU上进行
#log_device_placement,也是一个布尔型的参数，当它为True时，日志中将会记录每个节点被安排在哪个设备上以方便调试，而在生产环境中将这个参数设置为False可以减少
