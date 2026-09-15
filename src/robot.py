import numpy as np

class Robot:
    def __init__(self):
        pass

    def move_to_board(self):
        return np.array([
            [-0.5685,   0.82168,   -0.04076,  -365.341],
            [0.82176,   0.56951,    0.0192,     95.315],
            [0.03899,  -0.02257,   -0.99898,   509.752],
            [0,         0,          0,           1    ],
        ])

    def make_move(self, state, move):
        pass

    def quit(self):
        pass