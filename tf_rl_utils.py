import tensorflow as tf
#import tensorflow.contrib.slim as slim
import numpy as np


def discount_rewards(r, gamma = 0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def prepro(I):
    """ prepro 20x10 uint8 frame into 200 (20x10) 1D float vector """
    return I.astype(np.float).ravel()


#class Agent():
#    def __init__(self, lr, s_size, a_size, h_size):
#        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
#        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
#        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
#        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
#        # TODO max want to roll the dice
#        self.chosen_action = tf.argmax(self.output, 1)
#
#        # there will be one for EACH example in batch
#        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
#        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
#
#        # how many total outputs there are
#        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
#        # for each step, what was the output that we took
#        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
#        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)
#
#        tvars = tf.trainable_variables()  # the weights
#        # self.tvars = tvars
#        self.gradient_holders = []
#        for idx, var in enumerate(tvars):
#            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')  # why no shape
#            self.gradient_holders.append(placeholder)
#
#        # above block shouldn't affect
#        self.gradients = tf.gradients(self.loss, tvars)
#
#        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
