import json
import os
import sys

import numpy as np
import tensorflow as tf

import tetris_api
import tf_rl_utils

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

ACTIONS = {3: 'up', 2: 'right', 1: 'left', 0: 'down'}

if sys.version_info.major == 2:
    range = xrange

# tetris
num_cols = 10
num_rows = 20

# learning params
gamma = 0.9
learning_rate = 1e-2
max_ep = 99999 # how many steps to take
num_hidden = 25
update_frequency = 25 # after how many games to update model

# create environment
env = tetris_api.PyTetrisEnv()
tf.reset_default_graph()  # clear the Tensorflow graph.

sess = tf.InteractiveSession()

# build nodes
x = tf.placeholder(tf.float32, shape=[None, num_cols*num_rows])
actions = tf.placeholder(tf.int32, shape=[None])
rewards = tf.placeholder(tf.float32, shape=[None])

# 1st layer 
W1 = weight_variable([num_cols*num_rows, num_hidden])
b1 = bias_variable([num_hidden])
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
 
# output 
W2 = weight_variable([num_hidden, 4])
b2 = bias_variable([4])
y = tf.matmul(h1,W2) + b2
output = tf.nn.softmax(y)

# get prob for chosen action
# first need to figure out indices of all of the chose actions
# from the flattened array
action_indices = (tf.range(0, tf.shape(y)[0]) * tf.shape(y)[1]) + actions
prob_for_picked_actions = tf.gather(tf.reshape(output, [-1]), action_indices)

# loss function
cross_entropy = -tf.reduce_mean(tf.log(prob_for_picked_actions) * rewards)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# initialize variables
sess.run(tf.global_variables_initializer())


env = tetris_api.PyTetrisEnv()
all_ep_history = []
game_num = 0
game_rewards = []
action_history = []
while True:
    s_2D, _, _, _ = env.reset(number_of_rows=num_rows, number_of_cols=num_cols)
    s = tf_rl_utils.prepro(s_2D)
#     prior_state = np.zeros_like(s)
    ep_history = []
    for j in range(max_ep):
        # determine action
#         del_state = s - prior_state 
#         a_dist = sess.run(output, feed_dict={x: [del_state]})
        a_dist = sess.run(output, feed_dict={x: [s]})
        picked_action_prob = np.random.choice(a_dist[0], p=a_dist[0])
        action = np.argmax(a_dist == picked_action_prob)
        action_history.append(action)
        
        # take action
#         prior_state = s
        s_2D, r, done, _ = env.step(ACTIONS[action]) 
        s = tf_rl_utils.prepro(s_2D)
        
#         ep_history.append(np.array([del_state, action, r]))
        ep_history.append(np.array([s, action, r]))
        if done:
            # game_states = []
            ep_history = np.array(ep_history)
            r = tf_rl_utils.discount_rewards(ep_history[:, 2], gamma=gamma)
            # z-score of r
#             ep_history[:, 2] = (r - np.mean(r))/np.std(r)
            ep_history[:, 2] = r
            all_ep_history.append(ep_history)
            #TODO: make this raw rewards
            game_rewards.append(np.sum(r))
            ep_history = []

            its_time_to_update_weights = ((game_num % update_frequency) == 0 and (game_num != 0))
            if its_time_to_update_weights:
                print game_num, '{:2.2f}'.format(np.mean(game_rewards[game_num-update_frequency:game_num]))
                all_ep_history = np.vstack(all_ep_history)
                feed_dict = {rewards: all_ep_history[:, 2],
                             actions: all_ep_history[:, 1],
                             x: np.vstack(all_ep_history[:, 0])}
                sess.run(train_step, feed_dict=feed_dict)
                all_ep_history = []
                
            break
    game_num += 1
