"""
wrapper to make tetris.py have gym-like interface
"""
import copy
import sys

import numpy as np
import IPython

from python_tetris import python_tetris

if sys.version_info.major > 2:
    xrange = range

ACTIONS = {3: 'up', 2: 'right', 1: 'left', 0: 'down'}


def highest_piece(board, board_height=20):
    column_depths = np.array(board).argmax(axis=0)
    column_heights = board_height - column_depths[
        column_depths != 0]  # Exclude columns with no pieces.
    board_has_pieces = len(column_heights) > 0
    if board_has_pieces:
        return column_heights.max()
    else:  # No pieces on board yet.
        return 0


def _row_multiplier(r):
    """
    1 row: 1
    2 rows: 3 points
    3 rows: 7 points
    4 rows: 20 points """
    if r == 2:
        return 3
    if r == 3:
        return 7
    if r == 4:
        return 20
    return r


class PyTetrisEnv():
    def __init__(self, reward_blocked_cells=False):
        self.game = None
        self.old_piece = None  # Special case for first piece.
        self.old_height = None  # Special case for first piece.
        self.state = None
        self.reward = None
        self.cum_score = None

    def _process_reward(self, report, reward_factor=10):
        """Convert o to nparray, reward to float"""
        o, d, cum_score, reward = report.state,\
                                  report.done,\
                                  report.score,\
                                  report.score_from_action

        if reward > 0:
            reward = _row_multiplier(reward)

        return np.array(o), d, float(cum_score), float(reward)

        new_piece = (self.game.piece != self.old_piece)
        board_is_open = (self.old_height == 0)


#             reward *= reward_factor
#        if new_piece and not board_is_open:  # Didn't clear any rows.
#            # TODO(crawles): remove this reward shaping.
#            new_height = highest_piece(self.game.board,
#                                       self.game.number_of_rows)
#            reward = -(new_height - self.old_height) * 0.1

#        return np.array(o), d, float(cum_score), float(reward)

    def reset(self, **kwargs):
        self.game = python_tetris.Tetris(**kwargs)
        self.old_piece = None
        self.old_height = 0
        self.state, d, self.cum_score, self.reward = self._process_reward(
            self.game.step_forward('right'))
        return self.state

    def render(self):
        import os
        os.system('clear')
        print(self.game.pretty_print_board(self.state))
        print('Score: ', self.cum_score, " Reward: ", self.reward)

    def step(self, action):
        if isinstance(action, np.ndarray):
            assert action.shape == ()
            action = np.asscalar(action)
        self.old_piece = self.game.piece  # Special case for first piece.
        self.old_height = highest_piece(self.game.board,
                                        self.game.number_of_rows)

        self.state, d, self.cum_score, self.reward = self._process_reward(
            self.game.step_forward(ACTIONS[action]))
        if d:
            reward = -1

        return self.state, self.reward, d, None
