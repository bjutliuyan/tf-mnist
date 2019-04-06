'''
    Describe:   手写数字识别(一层感知机实现，利用Dropout、Adagrad优化，
                隐含层选用relu激活函数，输出层为softmax激活函数)
    Author:     liu yan
    Modify:     2019-04-06
    Accuracy:   0.978
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#读取数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#设置权重和偏置
in_units = 784                                                                  #输入节点数
h1_units = 300                                                                  #隐含层节点数
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))         #权重初始化为截断的正态分布，标准差为0.1
b1 = tf.Variable(tf.zeros([h1_units]))                                          #初始化隐藏层的偏置
W2 = tf.Variable(tf.zeros([h1_units, 10]))                                      #初始化输出层的权重
b2 = tf.Variable(tf.zeros([10]))                                                #初始化输出层的偏置

#构建模型
x = tf.placeholder(tf.float32, [None, in_units])
keep_pro = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)                                     #隐藏层
hidden1_drop = tf.nn.dropout(hidden1, keep_pro)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)                             #输出层

#定义loss,选择优化器,指定优化器优化loss
y_ =tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

#开始训练
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_pro: 0.75})

    #对模型进行准确率评测
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_pro: 1.0}))


