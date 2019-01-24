# 学习《Tensorflow 实战Google深度学习框架》
> 2019.1.24


## 第一章 深度学习简介：
* 主要讲tensorflow在google的应用

## 第二章 Tensorflow环境搭建
* 书中推荐依赖包Protocol Buffer和Bazel
* 书中推荐安装Docker和pip
* 个人安装conda+pip，交互式用jupyter notebook，可视化用tensorboard

## 第三章 Tensorflow 入门
* tensorflow计算模型-计算图
	** 有效的整理tensorflow程序中的资源也是计算图的一个重要功能
	** tf.add_to_collection 函数可以将资源加入一个或多个集合中，然后通过tf.get_collection获取一个集合里面的所有资源
	** 这里的资源可以是张量、变量或者运行程序所需要的队列资源
