import os
import sys

import click

import tetris

# TODO: this was to quickly, was impossible to play
steps_til_drop_gen = cycle(reversed(range(4)))
def step_forward(game, next_move):
    ''' keep track of number of moves to drop'''
    steps_till_drop = next(steps_til_drop_gen)
    report = game.move_piece(next_move)
    if report.done:
        return report
    elif steps_till_drop == 0: #automatic drop
        new_report = game.move_piece('down')
        if new_report.done:
            return new_report

    return report

def play_game(game = None):
    if game is None:
        game = tetris.Tetris()
    moves = {'\x1b[A': 'up', '\x1b[B': 'down', '\x1b[D': 'left', '\x1b[C': 'right'}
    for _ in range(50000):
        print(game.print_board())
        print(str.format("\n\n\nlines: {0}", game.total_lines))
        while 1:
            user_dir = click.getchar()
            if user_dir in moves:
                next_move = moves[user_dir]
                break
            elif user_dir == 'q':
                return game
            else:
                pass
        report = game.move_piece(next_move)
        if report.done:
            game.start()
        os.system('clear')

