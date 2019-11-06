'''
@description: 保存手写数字识别前20张图片示例
@date： 2019/11/03 22:23
@author：ly
'''

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# 如果没这个文件夹自动创建
save_dir = '../MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

for i in range(20):
    image_array = mnist.train.images[i, :]
    image_array = image_array.reshape(28, 28)
    #MNIST数据集图片灰度值都是0或者1，这里乘以256便于直观化图像
    image_array *= 256
    filename = save_dir + 'mnist_train_%d.jpg' % i
    image = Image.fromarray(image_array);
    if image.mode == "F":
        image = image.convert('RGB')
    image.save(filename)
