import chess
import numpy as np
from scipy.spatial.transform import Rotation
from src.constants import *

def fix_rotation_error(pose):
    rotation = Rotation.from_matrix(pose[:3, :3]).as_matrix()
    return make_pose(rotation, pose[:3, 3])


def make_pose(rotation, position):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.ravel(position)[:3]
    return pose


def rotation_vector_to_pose(rotation_vector, position):
    return make_pose(Rotation.from_rotvec(rotation_vector).as_matrix(), position)


def roll_pitch_yaw_to_pose(position, roll=0.0, pitch=0.0, yaw=0.0):
    rotation = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return make_pose(rotation, position)


def move_along_axis(pose, axis, distance):
    moved = pose.copy()
    moved[:3, 3] = pose[:3, 3] + pose[:3, axis] * distance
    return moved


def pixel_to_surface(pixel, surface_to_camera, height=0.0):
    ray = np.linalg.solve(CAMERA_MATRIX, [pixel[0], pixel[1], 1.0])
    normal = surface_to_camera[:3, 2]

    raised_origin = surface_to_camera[:3, 3] + normal * height
    hit = ray * (np.dot(normal, raised_origin) / np.dot(normal, ray))

    return np.linalg.inv(surface_to_camera) @ np.append(hit - normal * height, 1.0)


def camera_to_pixel(pose):
    pixel = CAMERA_MATRIX @ pose[:3, 3]
    pixel /= pixel[2]
    return (int(pixel[0]), int(pixel[1]))


def square_to_surface(square):
    half_board = 4 * SQUARE_SIZE
    offset = SQUARE_SIZE / 2 - half_board
    return np.array([
        (7 - chess.square_file(square)) * SQUARE_SIZE + offset,
        (7 - chess.square_rank(square)) * SQUARE_SIZE + offset,
        0.0,
        1.0,
    ])


def surface_to_square(pose):
    half_board = 4 * SQUARE_SIZE
    file_index = 7 - int((pose[0, 3] + half_board) // SQUARE_SIZE)
    rank_index = 7 - int((pose[1, 3] + half_board) // SQUARE_SIZE)

    if not (0 <= file_index < 8 and 0 <= rank_index < 8):
        return None

    return chess.square(file_index, rank_index)
