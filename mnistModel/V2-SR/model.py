
import tensorflow as tf
class NetWork:
    def __init__(self):
        self.weights = tf.Variable(tf.zeros([784, 10]))
        self.biases = tf.Variable(tf.zeros([10]))

        self.x = tf.placeholder("float", [None, 784])
        self.y_real = tf.placeholder("float", [None, 10])
        self.y = tf.nn.softmax(tf.matmul(self.x, self.weights) + self.biases)

        #选择交叉熵所谓损失函数
        self.loss = -tf.reduce_sum(self.y_real * tf.log(self.y + 1e-10))
        self.train = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_real, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

