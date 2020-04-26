import time

import jax
import jax.numpy as np
from flax import nn
from flax import optim

import numpy as onp
import tensorflow as tf

from functools import partial
import tetris_api
import tf_rl_utils
import utils

#TODO: move this into class
@jax.jit
def take_train_step(optimizer, ob_batch, ac_dist_batch, re_batch):
  """Train for a single step."""
  def loss_fn(model):
    _, logp, values = model(batch['image'])
    loss = cross_entropy_loss(logits, batch['label'])
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, batch['label'])
  return optimizer, metrics

class A2CAgent:

    def __init__(self, key, env, model, params):
        self.key = key
        self.env = env
        self.model = model
        self.params = params
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        optimizer_def = optim.Momentum(learning_rate=self.params['--learning_rate'],
                                       beta=self.params['--beta'])
        optimizer = optimizer_def.create(self.model)
        return optimizer

    def _sample_trajectory(self):
        """Play a game."""
        # TODO: Incorporate max_path_length, so we don't play for too long.
        with nn.stochastic(self.key):
            # List of lists for each game.
            obs, acs, logps, values, rewards = [], [], [], [], []
            ob = self.env.reset()
            done = False
            while not done:
                obs.append(ob)
                (a, logp), value = self.model(ob.reshape((ob.size,)),
                                              action_dim=self.env.n_actions)
                a = onp.asarray(a)  # JAX --> numpy
                acs.append(a)
                logps.append(logp[a])
                values.append(onp.asscalar(value[0]))
                ob, r, done, _ = self.env.step(a)
                rewards.append(r)
        return utils.path(obs, acs, logps, values, rewards)

    def _sample_trajectories(self):
        """Play games and collect at least `batch` trajectories."""
        print('Collecting data to use for training...')
        paths = []
        n_env_steps = 0
        while n_env_steps < self.params['--batch_size']:
            path = self._sample_trajectory()
            paths.append(path)
            n_env_steps += len(path.obs)
        return paths, n_env_steps

    def _calculate_q_vals(self, paths):
        """Monte Carlo estimation of the Q function."""
        return onp.concatenate([self._discounted_cumsum(p.rewards, self.params['--gamma']) for p in paths])

    @staticmethod
    def _discounted_cumsum(rewards, gamma):
        """Calculate rewards-to-go estimate of return at each step."""
        all_discounted_cumsums = []
        # For loop over steps (t) of the given rollout.
        for start_time_index in range(len(rewards)):
            # 1) create a list of indices (t'): goes from t to T-1.
            indices = onp.arange(start_time_index, len(rewards))
            # 2) create a list where the entry at each index (t') is gamma^(t'-t).
            discounts = gamma ** (indices - start_time_index)
            # 3) create a list where the entry at each index (t') is gamma^(t'-t) * r_{t'}.
            # Hint: remember that t' goes from t to T-1, so you should use the rewards from
            # those indices as well.
            discounted_rtg = rewards[indices] * discounts
            # 4) calculate a scalar: sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
            sum_discounted_rtg = discounted_rtg.sum()
            # appending each of these calculated sums into the list to return
            all_discounted_cumsums.append(sum_discounted_rtg)
        return onp.array(all_discounted_cumsums)

    @staticmethod
    def _estimate_advantage(pred_values, q_values):
        """Use state-dependent baseline to reduce variance."""
        b_n_unnormalized = pred_values
        b_n = b_n_unnormalized * onp.std(q_values) + onp.mean(q_values)
        adv_n = q_values - b_n
        # Normalize the resulting advantages.
        adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def take_train_step(self):
        paths, n_env_steps = self._sample_trajectories()
        q_values = self._calculate_q_vals(paths)
        advantages = self._estimate_advantage(onp.concatenate([p.values for p in paths]), q_values)
        return paths, n_env_steps

    def train(self):
        total_env_steps, total_n_games = 0, 0
        for train_step in range(self.params['--n_iter']):
            print('crcr taking train step')
            paths, n_env_steps = self.take_train_step()
            print('crcr took train step')
            total_env_steps += n_env_steps
            total_n_games += len(paths)
        print("Played ", total_n_games, " games for ", total_env_steps, " steps.")

    def update(self ):


    def test(self, render=True):
        with nn.stochastic(self.key):
            ob, done, ep_reward = self.env.reset(), False, 0
            while not done:
                (action, logp), _ = self.model(ob.reshape((obs.size, )),
                                               action_dim=self.env.n_actions)
                action = onp.asarray(action)  # JAX --> numpy
                if render:
                    time.sleep(0.1)
                ob, reward, done, _ = self.env.step(action)
                ep_reward += reward
                if render:
                    self.env.render()
        return ep_reward


#class oldA2CAgent:
#
#    def __init__(self, model):
#        self.model = model
#
#    def test(self, env, render=True):
#        obs, done, ep_reward = env.reset(), False, 0
#        while not done:
#            action, _ = self.model.get_action_value(utils.flatten(obs)[None, :])
#            obs, reward, done, _ = env.step(action)
#            ep_reward += reward
#            if render:
#                env.render()
#        return ep_reward


class FullyConnected(object):
    def __init__(self, graph, X, y, n_units, learning_rate):
        n_hidden1, n_hidden2, n_output = n_units
        with graph.as_default() as g:
            with tf.name_scope("dnn"):
                self.hidden1 = tf.layers.dense(X,
                                               n_hidden1,
                                               name="hidden1",
                                               activation=tf.nn.relu)
                self.hidden2 = tf.layers.dense(self.hidden1,
                                               n_hidden2,
                                               name="hidden2",
                                               activation=tf.nn.relu)
            self.logits = tf.layers.dense(self.hidden2,
                                          n_output,
                                          name="output")
            with tf.name_scope("loss"):
                self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y, logits=self.logits)
                self.loss = tf.reduce_mean(self.xentropy, name="loss")
            with tf.name_scope("train"):
                self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
                self.training_op = self.optimizer.minimize(self.loss)
