import jax
import jax.numpy as np

import numpy as onp
import tensorflow as tf
import tensorflow.keras.layers as kl

from flax import nn
from flax import optim

from jax import random




class MLPModel(nn.Module):
    def apply(self, x, action_dim):
        """Policy and value function model for Advantage Actor Critic."""
        # Actor
        x1 = nn.Dense(x, features=128)
        x1 = nn.relu(x1)
        x1 = nn.Dense(x1, features=action_dim)
        # Critic
        x2 = nn.Dense(x, features=128)
        x2 = nn.relu(x2)
        x2 = nn.Dense(x2, features=1)
        return x1, x2


class ActorCriticNetwork(nn.Module):
    def apply(self, x, action_dim=None):
        action_logits, state_logits = MLPModel(x, action_dim)
        # TODO: see if you can reuse the rng from this class.
        # This might help. https://github.com/google/flax/issues/179.
        # https://colab.research.google.com/drive/1eDXEVd8NPXgaSwn7jEMxsZTDVHNUtHYK
        action = jax.random.categorical(nn.make_rng(), action_logits)
        return (action, nn.log_softmax(action_logits)), state_logits

    @staticmethod
    def compute_actor_loss(logp, advantages):
        loss = np.log(logp) * advantages

    @staticmethod
    def compute_critic_loss(values, returns):
        # TODO: Monte Carlo and Bootstrap approach.
        loss = np.sqrt(np.mean(np.square(values - returns)))


#class MLPModel(tf.keras.Model):
#
#    def __init__(self, num_actions, n_layers = 1):
#        """Policy and value function model for Advantage Actor Critic."""
#        super().__init__(name='mlp_model')
#        self.num_actions = num_actions
#        self.hidden_layers = []
#        for _ in range(n_layers):
#            self.hidden_layers.append(kl.Dense(50, activation='relu'))
#        self.value = kl.Dense(1, activation=None, name='value')
#        self.logits = kl.Dense(num_actions, activation=None, name='policy_logits')
#
#    def call(self, inputs, **kwargs):
#        x = tf.convert_to_tensor(inputs)
#        # TODO(crawles): Experiment with not using the same network here.
#        for l in self.hidden_layers:
#            x = l(x)
#        return self.logits(x), self.value(x)
#
#    def _sample_action(self, logits):
#        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
#
