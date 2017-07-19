import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_rl_utils

from rl_tensorflow import python_tetris as pytetris_wrapper

ID = 26

num_cols = 6
num_rows = 16
gamma = 0.8
num_hidden = 50
learning_rate = 1e-2

max_ep = 99999999 # how many steps to take
update_frequency = 50

start_time = time.time()

params = "Tetris_{id}:  {num_rows} x {num_cols} , gamma: {gamma}, num_hidden: {num_hidden}, update_freq: {update_freq}".format(
    id = ID, num_rows = num_rows, num_cols = num_cols, gamma = gamma, num_hidden = num_hidden, update_freq = update_frequency,
    learning_rate = learning_rate
)

# results
save_dir = "./results/tetris_{id}/".format(id=ID)
save_wins = os.path.join(save_dir, 'wins')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
elif os.path.exists(save_wins):
    shutil.rmtree(save_wins)
os.makedirs(save_wins)

save_model_name = os.path.join(save_dir, "tetris{id}.ckpt".format(id=ID))
result_csv = os.path.join(save_dir, "total_reward{id}.csv".format(id=ID))
pd.DataFrame([params + ', time: init']).to_csv(os.path.join(save_dir, 'params_tetris{id}.txt').format(id=ID), header=None, index=False)

# create environment
env = pytetris_wrapper.PyTetrisEnv()
tf.reset_default_graph() # Clear the Tensorflow graph.

# create agent
myAgent = tf_rl_utils.agent(lr=learning_rate, s_size=num_cols*num_rows, a_size=4, h_size=num_hidden) #Load the agent.
tf.add_to_collection('output', myAgent.output)
tf.add_to_collection('state_in', myAgent.state_in)
all_saver = tf.train.Saver()

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
i = 0
total_reward = []
total_lenght = []
all_ep_history = []
prob_history = []

# keep track of wins
game_states = []

gradBuffer = sess.run(tf.trainable_variables())
for ix,grad in enumerate(gradBuffer):
    gradBuffer[ix] = grad * 0

print "\n"
print params
while True:
    s, _, _, _ = env.reset(number_of_rows=num_rows, number_of_cols=num_cols)
    s = tf_rl_utils.prepro(s)
    del_state = s - np.zeros_like(s)
    running_reward = 0
    ep_history = []
    for j in range(max_ep):
        a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in:[s]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        prob_history.append(a_dist[0])
        # np.vstack(prob_history).mean(axis=0)
        ACTIONS = {3: 'up', 2: 'right', 1: 'left', 0: 'down'}

        s1,r,d,_ = env.step(ACTIONS[a]) # Get our reward for taking an action given a bandit.
        game_states.append(env.game.print_board())
        s1 = tf_rl_utils.prepro(s1)

        ep_history.append([del_state, a, r, s1])
        s = s1
        running_reward += r
        if r > 0:
            json.dump(game_states,open(os.path.join(save_wins,'game_{}_step_{}.json'.format(i, j)),'wb'))
        if d == True:
            game_states = []
            ep_history = np.array(ep_history)
            ep_history[:, 2] = tf_rl_utils.discount_rewards(ep_history[:, 2], gamma=gamma)
            all_ep_history.append(ep_history)
            if i % update_frequency == 0 and i != 0:
                all_ep_history = np.vstack(all_ep_history)
                feed_dict = {myAgent.reward_holder: all_ep_history[:, 2],
                             myAgent.action_holder: all_ep_history[:, 1], myAgent.state_in: np.vstack(all_ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                all_ep_history = []
            total_reward.append(running_reward)
            total_lenght.append(j)
            break

    #Update our running tally of scores.
    if i % 10 == 0:
        np.savetxt(result_csv, total_reward, delimiter=',')
        prob_results = ['{}:{:0.5f}'.format(ACTIONS[ai],p) for ai,p in enumerate(np.vstack(prob_history).mean(axis=0))]
        print np.mean(total_reward[-10:]), len(total_reward), ','.join(prob_results)
        prob_history = []

    if i % 100 == 0:
        running_time = time.time() - start_time
        cur_stats = params + ', time: {}'.format(running_time)
        print 'saved model for model: ', cur_stats
        all_saver.save(sess, save_model_name)
        pd.DataFrame([params + ', time: {}'.format(cur_stats)]).to_csv(os.path.join(save_dir, 'params_tetris{id}.txt').format(id=ID),
                                                       header=None, index=False)

    i += 1