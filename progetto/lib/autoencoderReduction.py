from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble._hist_gradient_boosting.tests import test_loss
import numpy as np


class AutoencoderReduction:
    def __init__(self,n_inputs, n_hidden, n_outputs, learning_rate):
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, n_inputs])
        self.hidden = tf.compat.v1.layers.dense(self.X, n_hidden, activation=tf.nn.sigmoid)
        self.outputs = tf.compat.v1.layers.dense(self.hidden, n_outputs)

        reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        self.training_op = optimizer.minimize(reconstruction_loss)

        self.init = tf.compat.v1.global_variables_initializer()

    def fit_predict(self, training_set,test_set, epochs):
        with tf.compat.v1.Session() as sess:
            self.init.run()
            for iteration in range(epochs):
                self.training_op.run(feed_dict={self.X: training_set})

            codings_val = self.hidden.eval(feed_dict={self.X: test_set})
            rectruction_val = self.outputs.eval(feed_dict={self.X: test_set}) 
            #self.loss_test =tf.reduce_mean(tf.square(codings_val - self.X))
            #self.loss_test = loss_test.eval(feed_dict={self.X, test_set})

        return codings_val, rectruction_val
    
    def recostruction_error(self, X, X_cap):
        return (np.power(np.sum(np.power((X-X_cap),2), axis = 1),1/2)).values

   