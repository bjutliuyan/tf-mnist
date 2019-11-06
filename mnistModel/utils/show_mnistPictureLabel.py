'''
@description: 看前手写数字识别20张训练图片的label
@date： 2019/11/03 22:23
@author：ly
'''

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
#看前20张训练图片的label
for i in range(20):
    one_hot_label = mnist.train.labels[i, :]
    label = np.argmax(one_hot_label)
    print('mnist_train_%d.jpg label: %d' % (i, label))