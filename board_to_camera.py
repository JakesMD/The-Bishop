import cv2
import numpy as np

from constants import TAG_SIZE, TAG_SPACING, CAMERA_MATRIX, DIST_COEFFICIENTS

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

half_tag = TAG_SIZE / 2
half_center = TAG_SPACING / 2
tag_points = {}
for tag_id, (sign_x, sign_y) in enumerate([(-1, 1), (1, 1), (1, -1), (-1, -1)]):
    center_x, center_y = sign_x * half_center, sign_y * half_center
    tag_points[tag_id] = np.array([
        [center_x - half_tag, center_y + half_tag, 0.0],
        [center_x + half_tag, center_y + half_tag, 0.0],
        [center_x + half_tag, center_y - half_tag, 0.0],
        [center_x - half_tag, center_y - half_tag, 0.0],
    ])


def get_board_to_camera(frame: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    detected = {}
    for tag_corners, tag_id in zip(corners, ids.flatten()):
        if tag_id in tag_points:
            detected[tag_id] = tag_corners.reshape(4, 2)

    if not detected:
        return None

    if len(detected) == len(tag_points):
        square = np.array([detected[tag_id].mean(axis=0) for tag_id in tag_points], dtype=np.int32)
        cv2.polylines(frame, [square], isClosed=True, color=(0, 0, 255), thickness=3)

    object_points = np.concatenate([tag_points[tag_id] for tag_id in detected])
    image_points = np.concatenate(list(detected.values()))

    success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, CAMERA_MATRIX, DIST_COEFFICIENTS)

    if not success:
        return None

    board_to_camera = np.eye(4)
    board_to_camera[:3, :3], _ = cv2.Rodrigues(rotation_vector)
    board_to_camera[:3, 3] = translation_vector.flatten()

    return board_to_camera
