import os
import sys

import click

import tetris

def play_game():
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
                sys.exit()
            else:
                pass
        report = game.move_piece(next_move)
        if report.done:
            game.start()
        os.system('clear')