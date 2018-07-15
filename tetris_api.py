"""
wrapper to make tetris.py have gym-like interface
"""
import copy
import sys

import numpy as np

from python_tetris import python_tetris

if sys.version_info.major > 2:
    xrange = range


def highest_piece(board, board_height=20):
    column_depths = np.array(board).argmax(axis=0)
    column_heights = board_height-column_depths[column_depths != 0]  # Exclude columns with no pieces.
    board_has_pieces = len(column_heights) > 0
    if board_has_pieces:
        return column_heights.max()
    else:  # No pieces on board yet.
        return 0

class PyTetrisEnv():

    def __init__(self, reward_blocked_cells = False):
        self.game = None
        self.old_piece = None  # Special case for first piece.
        self.old_height = None  # Special case for first piece.
        


    def _process_reward(self, report, reward_factor=10):
        """Convert o to nparray, reward to float"""
        o, d, cum_score, reward = report.state,\
                                  report.done,\
                                  report.score,\
                                  report.score_from_action
        

        
        new_piece = (self.game.piece != self.old_piece)
        board_is_open = (self.old_height == 0)
        if reward > 0:
            reward *= reward_factor
        elif new_piece and not board_is_open:  # Didn't clear any rows.
            new_height = highest_piece(self.game.board, self.game.number_of_rows)
            reward = -(new_height - self.old_height)*0.1
        return np.array(o), d, float(cum_score), float(reward)

    def reset(self, **kwargs):
        self.game = python_tetris.Tetris(**kwargs)
        self.old_piece = None
        self.old_height = 0
        o, d, cum_score, reward = self._process_reward(self.game.step_forward( 'right'))
        return o, reward, d, None

    def step(self, action):
        self.old_piece = self.game.piece  # Special case for first piece.
        self.old_height = highest_piece(self.game.board, self.game.number_of_rows)
        
        o, d, cum_score, reward = self._process_reward(self.game.step_forward(action))
        if d:
            reward = -1

#        #DUMB reward
#        reward = int(action in ['left', 'right'])
#        if not reward:
#            reward = -1

        return o, reward, d, None
