"""
Usage:
  run_tetris.py <env> [--n_iter=<n_iter>] [--batch_size=<batch_size>] [--learning_rate=<lr>]\
                      [--beta=<opt_beta>] [--gamma=<gamma>]
  run_tetris.py -h | --help | --version

Options:
  --batch_size=<batch_size>    Batch size. [default: 512]
  --n_iter=<n_iter>            N training iterations. [default: 200]
  --learning_rate=<lr>         Learning rate. [default: 0.01]
  --beta=<opt_beta>            Beta. [default: 0.9]
  --gamma=<gamma>              Discount factor. [default: 0.99]

"""
import gym
from docopt import docopt
import tensorflow as tf

import agents
import models
import utils
import tetris_api

from flax import nn
import jax.numpy as np
from jax import random

arguments = docopt(__doc__, version='RLTetris')
utils.convert_types(arguments)
print(arguments)
if arguments['<env>'] == 'tetris':
    env = tetris_api.PyTetrisEnv()
    env.n_actions = 4
else:
    env = gym.make(arguments['env'])
    env.n_actions = env.action_space.n
env.reset()
input_shape = (1, ) + (env.reset().size, )
print('input_shape', input_shape)
rng = random.PRNGKey(0)  # Random number generator.

input_key, init_key, apply_key = random.split(random.PRNGKey(0), 3)
with nn.stochastic(random.PRNGKey(0)):
    _, initial_params = models.ActorCriticNetwork.init_by_shape(
        init_key, [(input_shape, np.float32)], action_dim=env.n_actions)
# _, initial_params = models.ActorCriticNetwork.init_by_shape(rng, [input_shape, np.float32])

model = nn.Model(models.ActorCriticNetwork, initial_params)
if __name__ == '__main__':
    agent = agents.A2CAgent(apply_key, env, model, arguments)
    # rewards_sum = agent.test(render=True)
    # print("Reward: ", rewards_sum)
    agent.train()
