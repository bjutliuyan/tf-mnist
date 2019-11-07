import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import NetWork


class Train:
    '''
    @description: 初始化训练模型
    '''
    def __init__(self):
        self.net = NetWork()
        self.mnist_data = input_data.read_data_sets('../MNIST_data/', one_hot=True)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    '''
    @description: 训练网络
    '''
    def train(self):
        batch_size = 100
        train_step = 3000
        for i in range(train_step):
            batch_xs, batch_ys = self.mnist_data.train.next_batch(batch_size)
            _, loss = self.sess.run([self.net.train, self.net.loss],
                                    feed_dict={self.net.x: batch_xs, self.net.y_real: batch_ys})
            if (i + 1) % 100 == 0:
                print('第%5d步，当前loss：%.2f' % (i + 1, loss))

    '''
    @description: 验证测试集准确率
    '''
    def calculate_accuracy(self):
        test_x = self.mnist_data.test.images
        test_label = self.mnist_data.test.labels
        accuracy = self.sess.run(self.net.accuracy, feed_dict={self.net.x:test_x,self.net.y_real:test_label})
        print("准确率: %.3f，共测试了%d张图片 " % (accuracy, len(test_label)))

if __name__ == "__main__":
    app = Train()
    #对训练集进行训练
    app.train()
    #对测试机进行测试
    app.calculate_accuracy()
