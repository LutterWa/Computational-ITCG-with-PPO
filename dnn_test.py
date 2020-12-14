import scipy.io
import numpy as np
import tensorflow.compat.v1 as tf
from dnn_regression import load_data, init_network

tf.disable_eager_execution()

class DeepNeuralNetwork:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x, y, trainer, loss = init_network()
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.y = self.sess.graph.get_tensor_by_name(name='layers/n_outputs/BiasAdd:0')
                saver = tf.train.Saver()
                checkpoint_path = tf.train.latest_checkpoint('./dnn_model')
                saver.restore(self.sess, checkpoint_path)

    def predict(self, state):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                return self.y.eval(feed_dict={self.x: state})


if __name__ == '__main__':
    dnn = DeepNeuralNetwork()
    x_train, y_train, x_test, y_test, x_vali, y_vali, train_set_size = load_data(shuffle_flag=False)

    dict_DNN = {'y_hat': dnn.predict(x_train[:int(1e6), :]), 'y':y_train[:int(1e6), :]}
    dataDNN = r'dnn_test.mat'
    scipy.io.savemat(dataDNN, dict_DNN)
