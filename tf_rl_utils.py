import sys
import time

import tensorflow as tf
import numpy as np

if sys.version_info.major > 2:
    xrange = range
    
ACTIONS = {3: 'up', 2: 'right', 1: 'left', 0: 'down'}
    

def discount_rewards(r, gamma = 0.99):
    """ Source: Karpathy
    Take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def prepro(I):
    """ prepro 20x10 uint8 frame into 200 (20x10) 1D float vector """
    return I.astype(np.float).ravel()

def variable_summaries(var, name):
    """https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
    Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        nonzero = tf.count_nonzero(var, dtype=tf.int32)
        per_nonzero = nonzero/tf.size(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('per_nonzero', per_nonzero)
        
        tf.summary.histogram('histogram', var)


def update_input(prior_states, new_state):
    """Pop old state, add new one."""
    prior_states.pop(0)
    prior_states.append(new_state)
    return np.hstack(prior_states)




def chose_from_action_dist(a_dist):
    """From the NN output, draw an action from the outputted action distribution"""
    picked_action_prob = np.random.choice(a_dist, p=a_dist)
    return np.argmax(a_dist == picked_action_prob)

def play_game(env,
              agent_step,
              max_ep,
              num_rows,
              num_cols,
              n_prior_states,
              sess=None):
    batch_start_time = time.time()
    batch_num_moves = 0

    s_2D, _, _, _ = env.reset(number_of_rows=num_rows, number_of_cols=num_cols)
    s = prepro(s_2D)
    prior_state = np.zeros_like(s)
    # init prior states
    prior_states = [prior_state] * n_prior_states
    action_history = []
    ep_history = []
    for j in range(max_ep):
        batch_num_moves += 1
        nn_input = update_input(prior_states, s)
        
        # Determine action.
        action_distribution = agent_step(nn_input)
        action = chose_from_action_dist(action_distribution)
        action_history.append(action)

        # Take action.
        prior_state = s
        s_2D, r, done, _ = env.step(ACTIONS[action]) 
        s = prepro(s_2D)

        ep_history.append(np.array([nn_input, action, r]))
        if done:
            break
    return ep_history, {'action_history': action_history}