import json
import os
import sys

import numpy as np
import tensorflow as tf

import tetris_api
import tf_rl_utils


ACTIONS = {3: 'up', 2: 'right', 1: 'left', 0: 'down'}

if sys.version_info.major == 2:
    range = xrange

ID = 26

num_cols = 6
num_rows = 16
gamma = 0.8
num_hidden = 50
learning_rate = 1e-2

max_ep = 99999999 # how many steps to take
update_frequency = 50 # after how many games to update model

params = "Tetris_{id}:  {num_rows} x {num_cols} , gamma: {gamma}, num_hidden: {num_hidden}, update_freq: {update_freq}".format(
    id = ID, num_rows = num_rows, num_cols = num_cols, gamma = gamma, num_hidden = num_hidden, update_freq = update_frequency,
    learning_rate = learning_rate
)

# create environment
env = tetris_api.PyTetrisEnv()
tf.reset_default_graph()  # clear the Tensorflow graph.

# create agent
myAgent = tf_rl_utils.Agent(lr=learning_rate, s_size=num_cols*num_rows, a_size=4, h_size=num_hidden)
tf.add_to_collection('output', myAgent.output)
tf.add_to_collection('state_in', myAgent.state_in)
all_saver = tf.train.Saver()

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
i = 0
total_reward = []
total_length = []
all_ep_history = []
# prob_history = []

# keep track of wins
# game_states = []

gradBuffer = sess.run(tf.trainable_variables())
for ix, grad in enumerate(gradBuffer):
    gradBuffer[ix] = grad * 0

while True:
    s, _, _, _ = env.reset(number_of_rows=num_rows, number_of_cols=num_cols)
    s = tf_rl_utils.prepro(s)
    del_state = s - np.zeros_like(s)
    running_reward = 0
    ep_history = []
    for j in range(max_ep):
        a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
        # roll the dice
        a = np.random.choice(a_dist[0], p=a_dist[0])
        # or pick max
        a = np.argmax(a_dist == a)
        # prob_history.append(a_dist[0])

        # Get our reward for taking an action
        s1, r, done, _ = env.step(ACTIONS[a]) 
        # game_states.append(env.game.print_board())
        s1 = tf_rl_utils.prepro(s1)

        ep_history.append([del_state, a, r, s1])
        # update state
        s = s1
        running_reward += r
        if done:
            # game_states = []
            ep_history = np.array(ep_history)
            ep_history[:, 2] = tf_rl_utils.discount_rewards(ep_history[:, 2], gamma=gamma)
            all_ep_history.append(ep_history)

            time_to_update_weights = ((i % update_frequency) == 0 and (i != 0))
            if time_to_update_weights:
                all_ep_history = np.vstack(all_ep_history)
                feed_dict = {myAgent.reward_holder: all_ep_history[:, 2],
                             myAgent.action_holder: all_ep_history[:, 1],
                             myAgent.state_in: np.vstack(all_ep_history[:, 0])}
                #TODO: ^ why vstack on del_state?

                # why not just one gradient, as in SGD?
                # maybe it is just one gradient for each weight
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)

                # clear gradients
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                all_ep_history = []
            total_reward.append(running_reward)
            total_length.append(j)
            break
    i += 1
