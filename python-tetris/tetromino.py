from random import randrange

class Tetromino(object):
    def __init__(self, rotations, current, x, y, shape):
        self.rotations = rotations
        self.current = current
        self.x = x
        self.y = y
        self.shape = shape
 
    @staticmethod
    def random(x=0, y=0):
        p = randrange(0, 7)

        if p == 0:
          return Tetromino.T(x, y)
        elif p == 1:
          return Tetromino.J(x, y)
        elif p == 2:
          return Tetromino.L(x, y)
        elif p == 3:
          return Tetromino.Z(x, y)
        elif p == 4:
          return Tetromino.S(x, y)
        elif p == 5:
          return Tetromino.Block(x, y)
        elif p == 6:
          return Tetromino.Line(x, y)

    @staticmethod
    def T(x, y):
        return Tetromino(
                rotations=[
                    [
                        [0, 1, 0],
                        [1, 1, 1]
                        ],
                    [
                        [1, 0],
                        [1, 1],
                        [1, 0]
                        ],
                    [
                        [1, 1, 1],
                        [0, 1, 0],
                        ],
                    [
                        [0, 1],
                        [1, 1],
                        [0, 1]
                        ]
                    ],
                current=0,
                x=x,
                y=y,
                shape='T'
                )

    @staticmethod
    def J(x, y):
        return Tetromino(
                rotations=[
                    [
                        [0, 1],
                        [0, 1],
                        [1, 1]
                        ],
                    [
                        [1, 0, 0],
                        [1, 1, 1],
                        ],
                    [
                        [1, 1],
                        [1, 0],
                        [1, 0],
                        ],
                    [
                        [1, 1, 1],
                        [0, 0, 1],
                        ]
                    ],
                current=0,
                x=x,
                y=y,
                shape='J'
                )

    @staticmethod
    def L(x, y):
        return Tetromino(
                rotations=[[
                    [1, 0],
                    [1, 0],
                    [1, 1]
                    ],
                [
                    [1, 1, 1],
                    [1, 0, 0]
                    ],
                [
                    [1, 1],
                    [0, 1],
                    [0, 1]
                    ],
                [
                    [0, 0, 1],
                    [1, 1, 1]
                    ]],
                current=0,
                x=x,
                y=y,
                shape='L'
                )


    @staticmethod
    def Z(x, y):
        return Tetromino(
                rotations=[[
                    [0, 1],
                    [1, 1],
                    [1, 0],
                    ],
                [
                    [1, 1, 0],
                    [0, 1, 1],
                    ],
                [
                    [0, 1],
                    [1, 1],
                    [1, 0]
                    ],
                [
                    [1, 1, 0],
                    [0, 1, 1],
                    ]], 
                current=0,
                x=x,
                y=y,
                shape='Z'
                )


    @staticmethod
    def S(x, y):
        return Tetromino(
                rotations=[[
                    [1, 0],
                    [1, 1],
                    [0, 1],
                    ],
                [
                    [0, 1, 1],
                    [1, 1, 0],
                    ],
                [
                    [1, 0],
                    [1, 1],
                    [0, 1],
                    ],
                [
                    [0, 1, 1],
                    [1, 1, 0],
                    ]], 
                current=0,
                x=x,
                y=y,
                shape='S'
                )

    @staticmethod
    def Block(x, y):
        return Tetromino(
                rotations=[[
                    [1, 1],
                    [1, 1],
                    ],
                [
                    [1, 1],
                    [1, 1],
                    ],
                [
                    [1, 1],
                    [1, 1],
                    ],
                [
                    [1, 1],
                    [1, 1],
                    ]],
                current=0,
                x=x,
                y=y,
                shape='Block'
                )


    @staticmethod
    def Line(x, y):
        return Tetromino(
                rotations=[[
                    [1],
                    [1],
                    [1],
                    [1]
                    ],
                [
                    [1, 1, 1, 1],
                    ],
                [
                    [1],
                    [1],
                    [1],
                    [1]
                    ],
                [
                    [1, 1, 1, 1],
                    ]],
                current=0,
                x=4,
                y=0,
                shape='Line'
                )



