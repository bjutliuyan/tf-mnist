import tensorflow as tf
import cv2
from model import NetWork
import numpy as np

CKPT_DIR = '../ckpt'

class Predict:
    def __init__(self):
        self.net = NetWork()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.restor()

    def restor(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError("未保存模型")

    def predict(self, image_path):
        # 读取图片
        img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 并转换为28 * 28
        res = cv2.resize(img_gray, (28,28), interpolation=cv2.INTER_LINEAR)
        flatten_img = np.reshape(res, 784)
        # 置为MNIST类的黑白图
        x = np.array([1 - flatten_img])
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})
        print('        -> Predict digit', np.argmax(y[0]))

if __name__ == "__main__":
    app = Predict()
    app.predict('../test_images/8.jpg')