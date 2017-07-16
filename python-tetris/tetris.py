from tetromino import Tetromino
from time import sleep
import os
import sys
from random import randrange
from collections import namedtuple


import numpy as np

ActionReport = namedtuple("ActionReport", "state done score score_from_action did_perform_move")

class Tetris(object):
    def __init__(self, number_of_rows=16, number_of_cols=10):
        self.number_of_rows = number_of_rows
        self.number_of_cols = number_of_cols
        self.start()

    def start(self):
        self.board = self.empty_board()
        self.piece = self.random_piece()
        self.lines_scored = []
        self.total_lines = 0
        self.moves = 0
        self.is_running = True

    def empty_board(self):
        return list(np.zeros([self.number_of_rows, self.number_of_cols], dtype = np.int))

    def combine_game_state(self):
        piece_indices = []
        for row_i, row in enumerate(self.piece.rotations[self.piece.current]):
            for col_i, col in enumerate(row):
                if col == 1:
                    piece_indices.append((col_i + self.piece.x, row_i + self.piece.y))

        combined_state=[]
        for row_i, row in enumerate(self.board):
            combined_row = []
            for col_i, col in enumerate(row):
                if (col_i, row_i) in piece_indices:
                    combined_row.append(1)
                else:
                    combined_row.append(col)

            combined_state.append(combined_row)

        return combined_state

    def print_board(self):
        output = ""

        for row in self.combine_game_state():
            output += "\n|"
            for col in row:
                if col:
                    output += '\u25A7'
                else: output += " "
                output += " "
            output += "|"

        output += "\n"
        for _ in range(2 + (self.number_of_cols * 2)):
            output += "_"

        return output

    def freeze_current_piece(self):
        self.board = np.array(self.board)

        # freeze piece on board
        piece = np.array(self.piece.rotations[self.piece.current])
        p_y, p_x = self.piece.y, self.piece.x
        p_height, p_width = piece.shape
        p_start_y, p_end_y = p_y, p_y + p_height
        p_start_x, p_end_x = p_x, p_x + p_width
        self.board[p_start_y:p_end_y, p_start_x:p_end_x] += piece

        # clear lines if needed
        row_is_cleared = self.board.sum(axis=1) == self.number_of_cols
        num_rows_cleared = np.sum(row_is_cleared)
        new_board = np.zeros_like(self.board)
        new_board[num_rows_cleared:, :] = self.board[~row_is_cleared]
        self.board = new_board

        if num_rows_cleared > 0:
            self.total_lines += num_rows_cleared
            self.lines_scored.append(num_rows_cleared)

        self.piece = self.random_piece()

        return num_rows_cleared


    def random_piece(self):
        return Tetromino.random(
                y=0,
                x=randrange(0, self.number_of_cols - 2)
                )

    def rotate_piece(self, kick_offset=0):
        new_current = (self.piece.current + 1) % len(self.piece.rotations)

        x, y = self.piece.x + kick_offset, self.piece.y

        for row_i, row in enumerate(self.piece.rotations[new_current]):
            for col_i, col in enumerate(row):
                if col == 1:
                    if row_i + y >= len(self.board):
                        return False

                    if col_i + x < 0:
                        return False

                    if col_i + x >= len(self.board[0]):
                        return self.rotate_piece(kick_offset=kick_offset - 1)


        for row_i, row in enumerate(self.piece.rotations[new_current]):
            for col_i, col in enumerate(row):
                if col == 1:

                    if row_i >= len(self.board) - 1:
                        return False

                    if self.board[row_i + y][col_i + x] == 1:
                        return False

        self.piece.x, self.piece.y = x, y
        self.piece.current = new_current

        return True

    def move_piece(self, direction):
        if not self.is_running:
            raise Exception("You must start a game before you can move")

        x, y = self.piece.x, self.piece.y

        if direction == 'up':
            self.rotate_piece()
        elif direction == 'left':
            x -= 1
        elif direction == 'right':
            x += 1
        elif direction == 'down':
            y += 1
        else:
            raise Exception(direction + " is not a valid command")

        for row_i, row in enumerate(self.piece.rotations[self.piece.current]):
            for col_i, col in enumerate(row):
                if col == 1:
                    if row_i + y >= len(self.board):
                        score_from_freeze = self.freeze_current_piece()
                        self.moves += 1
                        return ActionReport(state=self.combine_game_state(), done=False, score=self.total_lines, score_from_action=score_from_freeze, did_perform_move=False)

                    if col_i + x < 0:
                        self.moves += 1
                        return ActionReport(state=self.combine_game_state(), done=False, score=self.total_lines, score_from_action=0, did_perform_move=False)

                    if col_i + x >= len(self.board[0]):
                        self.moves += 1
                        return ActionReport(state=self.combine_game_state(), done=False, score=self.total_lines, score_from_action=0, did_perform_move=False)


        for row_i, row in enumerate(self.piece.rotations[self.piece.current]):
            for col_i, col in enumerate(row):
                if col == 1:
                    if row_i >= len(self.board) - 1:
                        self.moves += 1
                        return ActionReport(state=self.combine_game_state(), piece=self.piece, done=False, score=self.total_lines, score_from_action=0, did_perform_move=False)

                    if self.board[row_i + y][col_i + x] != 0:
                        if self.piece.y == 0:
                            self.is_running = False
                            self.moves += 1
                            return ActionReport(state=self.combine_game_state(), done=True, score=self.total_lines, score_from_action=0, did_perform_move=False)
                        elif direction == 'down':
                            score_from_freeze = self.freeze_current_piece()
                            self.moves += 1
                            return ActionReport(state=self.combine_game_state(), done=False, score=self.total_lines, score_from_action=score_from_freeze, did_perform_move=True)
                        else:
                            self.moves += 1
                            return ActionReport(state=self.combine_game_state(), done=False, score=self.total_lines, score_from_action=0, did_perform_move=True)


        self.piece.x, self.piece.y = x, y

        self.moves += 1
        return ActionReport(state=self.combine_game_state(), done=False, score=self.total_lines, score_from_action=0, did_perform_move=True)





if __name__ == "__main__":
    scores = []
    game = Tetris()

    moves = [
            'left', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right', 'right', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right',
    ]
    for _ in range(50000):
        for move in moves:
            print(game.print_board())
            print(str.format("\n\n\nlines: {0}", game.total_lines))
            sleep(0.05)

            next_move = move
            if next_move:
                report = game.move_piece(next_move)
                if report.done:
                    game.start()

            report = game.move_piece('down')
            if report.done:
                game.start()

            os.system('clear')
    sys.exit(0)

