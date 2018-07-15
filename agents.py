import json
import os
import sys

import numpy as np
import tensorflow as tf

import tetris_api
import tf_rl_utils

class FullyConnected(object):

	def __init__(self, graph, X, y, n_units, learning_rate):
		n_hidden1, n_hidden2, n_output = n_units
		# 1st layer 
		with graph.as_default() as g:
			with tf.name_scope("dnn"):
			    self.hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
			                              activation=tf.nn.relu)
			    self.hidden2 = tf.layers.dense(self.hidden1, n_hidden2, name="hidden2",
			                              activation=tf.nn.relu)
			    self.logits = tf.layers.dense(self.hidden2, n_output, name="output")
			with tf.name_scope("loss"):
				self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
				self.loss = tf.reduce_mean(self.xentropy, name="loss")
			with tf.name_scope("train"):
				self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
				self.training_op = self.optimizer.minimize(self.loss)