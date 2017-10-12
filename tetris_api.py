"""
wrapper to make tetris.py have gym-like interface
"""
import numpy as np

from python_tetris import python_tetris

class PyTetrisEnv():

    def __init__(self, reward_blocked_cells = False):
        self.game = None

    def _process_reward(self, report, reward_factor=10):
        """Convert o to nparray, reward to float"""
        o, d, cum_score, reward = report.state,\
                                  report.done,\
                                  report.score,\
                                  report.score_from_action
        if reward > 0:
            reward *= reward_factor

        return np.array(o), d, float(cum_score), float(reward)

    def reset(self, **kwargs):
        self.game = python_tetris.Tetris(**kwargs)
        o, d, cum_score, reward = self._process_reward(self.game.step_forward( 'right'))
        return o, reward, d, None

    def step(self, action):
        o, d, cum_score, reward = self._process_reward(self.game.step_forward(action))
        if d:
            reward = -1

#        #DUMB reward
#        reward = int(action in ['left', 'right'])
#        if not reward:
#            reward = -1

        return o, reward, d, None
