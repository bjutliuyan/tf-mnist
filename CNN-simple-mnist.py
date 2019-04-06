'''
    Describe:   手写数字识别(CNN实现)
                    C两层卷积层          [5, 5, 1, 32]  [5, 5, 32, 64]
                    一层池化层           [-1 2, 2, 1]
                    利用Dropout;         (0.5)
                    Adam优化;
                    隐含层选用relu激活函数;
                    输出层为softmax激活函数)
    Author:     liu yan
    Modify:     2019-04-06
    Accuracy:   0.992
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#1.读取数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#2.定义方法
'''定义权重创建方法'''
def weight_varialbe(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
'''定义偏置创建方法'''
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
'''定义卷积层创建方法'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
'''定义池化层创建方法'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#3.设置权重和偏置
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

#4.构建模型
W_conv1 = weight_varialbe([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)                    #定义第一个卷积层，5*5卷积核,channel为1,32个不同的卷积核
h_pool1 = max_pool_2x2(h_conv1)                                             #定义第一个池化层

W_conv2 = weight_varialbe([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)                    #定义第二个卷积层，5*5卷积核,channel为1,32个不同的卷积核
h_pool2 = max_pool_2x2(h_conv2)                                             #定义第二个池化层

W_fc1 = weight_varialbe([7 * 7 * 64, 1024])                                 #定义一个全连接层
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_pro = tf.placeholder(tf.float32)                                       #引入Dropout随机丢弃一部分节点数据来减轻过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_pro)

W_fc2 = weight_varialbe([1024, 10])                                         #Dropout层的输出连接一个softmax激活函数
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#定义loss,选择优化器,指定优化器优化loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#开始训练
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_pro: 0.5})

    # 对模型进行准确率评测
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_pro: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

