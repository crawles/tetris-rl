"""
wrapper to make tetris.py have gym-like interface
"""
import numpy as np

from python_tetris import python_tetris

class PyTetrisEnv():

    def __init__(self, reward_blocked_cells = False):
        self.game = None
        self.reward_blocked_cells = reward_blocked_cells
        if reward_blocked_cells:
            self.prev_blocked_cells = 0

    def _process_reward(self, report):
        """Convert o to nparray, reward to float"""
        o, d, cum_score, reward = report.state,\
                                  report.done,\
                                  report.score,\
                                  report.score_from_action
        reward_multiplier = 10
        if reward > 0:
            reward *= reward_multiplier
        if self.reward_blocked_cells:
            blocked_cells = self._board_num_blocked_cells()
            del_blocked_cells = blocked_cells - self.prev_blocked_cells
            reward += (-del_blocked_cells) * .1
            self.prev_blocked_cells = blocked_cells
        return np.array(o), d, float(cum_score), float(reward)

    def reset(self, **kwargs):
        self.game = python_tetris.Tetris(**kwargs)
        o, d, cum_score, reward = self._process_reward(self.game.step_forward( 'right'))
        return o, reward, d, None

    def step(self, action):
        o, d, cum_score, reward = self._process_reward(self.game.step_forward(action))
        if d:
            reward = -1
        # time.sleep(.05)
        # print self.game.print_board()
        return o, reward, d, None

    def _column_num_blocked_cells(self, col):
        num_empty = 0
        num_blocked = 0
        for v in col:
            num_empty += int(v == 0)
            if num_empty and v:
                num_blocked += num_empty
                num_empty = 0
        return (num_blocked)

    def _board_num_blocked_cells(self):
        num_blocked = 0
        for c in np.array(self.game.board).T:
            num_blocked += self._column_num_blocked_cells(reversed(c))
        return num_blocked