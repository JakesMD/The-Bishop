import cv2
import numpy as np
from constants import CAMERA_MATRIX, CAMERA_TO_FLANGE


def get_pixel_coordinates(frame, pixel, flange_to_base, board_to_camera):
    board_rotation = board_to_camera[:3, :3]
    board_translation = board_to_camera[:3, 3]

    x_norm, y_norm, _ = np.linalg.inv(CAMERA_MATRIX) @ np.array([pixel[0], pixel[1], 1])

    board_x, board_y, camera_z = np.linalg.inv(
            np.array([
                [-board_rotation[0][0], -board_rotation[0][1], x_norm],
                [-board_rotation[1][0], -board_rotation[1][1], y_norm],
                [-board_rotation[2][0], -board_rotation[2][1], 1],
            ])
        ) @ board_translation

    point_x = x_norm * camera_z
    point_y = y_norm * camera_z

    point_in_camera = np.array([point_x, point_y, camera_z, 1.0])
    point_in_base = flange_to_base @ CAMERA_TO_FLANGE @ point_in_camera

    base_point = point_in_base[:3]
    board_point = np.array([board_x, board_y])

    origin = (round(pixel[0]), round(pixel[1]))
    cv2.putText(
        img=frame,
        text=f"base: ({base_point[0]:.1f}, {base_point[1]:.1f}, {base_point[2]:.1f})",
        org=origin,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 255, 0),
        thickness=1
    )
    cv2.putText(
        img=frame,
        text=f"board: ({board_point[0]:.1f}, {board_point[1]:.1f})",
        org=(origin[0], origin[1] + 15),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 255, 0),
        thickness=1
    )

    return base_point, board_point
