import chess
import numpy as np

CAMERA_INDEX = 0

CAMERA_MATRIX = np.array([
    [965.1194082128692, 0.0, 627.6160586327128],
    [0.0, 962.9309950770537, 357.733973804143],
    [0.0, 0.0, 1.0],
])

DIST_COEFFICIENTS = np.array([
    0.15032469444219987,
    -0.3816527043247342,
    -0.0021951407170828635,
    -6.873027488957336e-05,
    0.3071576256549531,
])

CAMERA_TO_FLANGE = np.array([
    [0.9997486456989974,    -0.022144929796334926,  0.0034996438871934153,   0.7168662387680769],
    [0.021871770426231923,   0.9976581728030458,    0.0648058322816019,     -94.93275462451912 ],
    [-0.004926568932227251, -0.06471299964925373,   0.9978917559509913,     -82.99156933584061 ],
    [0.0,                   0.0,                    0.0,                     1.0               ],
])


ROBOT_APPROACH_HEIGHT = 60.0
ROBOT_GRIP_HEIGHT = 12.0

SQUARE_SIZE = 35
SURFACE_TAG_SIZE = 13.0
SURFACE_TAG_SPAN = np.array([612.5-15, 280-30-15])

PIECE_HEIGHT = 16
PIECE_DIAMETER = 35.0

PARKING_BOARD_MARGIN = 20.0
PARKING_TAG_MARGIN = 2.0
PARKING_PIECE_CLEARANCE = 2.0
PARKING_RESOLUTION = 2.0


def _surface_tag_points(corner):
    center_x, center_y = SURFACE_TAG_SPAN / 2 * corner
    half_tag = SURFACE_TAG_SIZE / 2
    return np.array([
        [center_x - half_tag, center_y + half_tag, 0.0],
        [center_x + half_tag, center_y + half_tag, 0.0],
        [center_x + half_tag, center_y - half_tag, 0.0],
        [center_x - half_tag, center_y - half_tag, 0.0],
    ])

SURFACE_TAG_POINTS = {
    0: _surface_tag_points([-1,  1]),
    1: _surface_tag_points([ 1,  1]),
    2: _surface_tag_points([ 1, -1]),
    3: _surface_tag_points([-1, -1]),
}

PIECE_IDS = {
    10: chess.Piece(chess.PAWN, chess.WHITE),
    11: chess.Piece(chess.ROOK, chess.WHITE),
    12: chess.Piece(chess.KNIGHT, chess.WHITE),
    13: chess.Piece(chess.BISHOP, chess.WHITE),
    14: chess.Piece(chess.QUEEN, chess.WHITE),
    15: chess.Piece(chess.KING, chess.WHITE),
    20: chess.Piece(chess.PAWN, chess.BLACK),
    21: chess.Piece(chess.ROOK, chess.BLACK),
    22: chess.Piece(chess.KNIGHT, chess.BLACK),
    23: chess.Piece(chess.BISHOP, chess.BLACK),
    24: chess.Piece(chess.QUEEN, chess.BLACK),
    25: chess.Piece(chess.KING, chess.BLACK),
}