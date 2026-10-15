import numpy as np

CAMERA_INDEX = 0

CAMERA_MATRIX = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0],
])

DIST_COEFFICIENTS = np.array([
    0,
    0,
    0,
    0,
    0,
])

CAMERA_TO_FLANGE = np.array([
    [1.0, 0.0, 0.0, 10.0],
    [0.0, 1.0, 0.0, 10.0],
    [0.0, 0.0, 1.0, 10.0],
    [0.0, 0.0, 0.0,  1.0],
])


ROBOT_APPROACH_HEIGHT = 15
ROBOT_GRIP_HEIGHT = 5

SQUARE_SIZE = 35
SURFACE_TAG_SIZE = 15.0
SURFACE_TAG_SPAN = np.array([589.5, 267])

PIECE_HEIGHT = 6
PIECE_DIAMETER = 30

ICON_CROP_SIZE_PX = 64
ICON_CROP_MARGIN_RATIO = 1.25

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

TOTAL_PIECES = 34 # 16 white, 16 black + 2 extra queens