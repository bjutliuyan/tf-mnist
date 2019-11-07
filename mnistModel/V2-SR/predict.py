import tensorflow as tf
import cv2
from model import NetWork
from PIL import Image
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
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img, 784)
        x = np.array([1 - flatten_img])
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})
        print('        -> Predict digit', np.argmax(y[0]))

if __name__ == "__main__":
    app = Predict()
    app.predict('../test_images/4.jpg')