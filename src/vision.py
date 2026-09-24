import chess
import cv2
import numpy as np
from src.types import *
from src.board import *
from src.constants import *
from src.transforms import *


class Vision:
    def __init__(self):
        self.capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        detector_parameters = cv2.aruco.DetectorParameters()
        detector_parameters.detectInvertedMarker = True
        self.tag_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5),
            detector_parameters
        )

    def detect_state(self, game_board: chess.Board):
        self.capture.grab()
        successful, frame = self.capture.read()
        if not successful:
            raise RuntimeError("Couldn't capture image")
 
     #   frame = cv2.rotate(frame, cv2.ROTATE_180)
        tags = self._detect_tags(frame)
        surface_to_camera = self._get_surface_to_camera(tags)

        if surface_to_camera is None:
            return DetectedState(frame=frame, board=None, pieces=None, parking_spot=None, surface_to_camera=None)

        pieces = self._get_pieces(frame, tags, surface_to_camera)
        parking_spot = self._get_parking_spot(frame, pieces, surface_to_camera)

        detected_board = set_board_pieces(game_board, pieces)

        return DetectedState(frame=frame, board=detected_board, pieces=pieces, parking_spot=parking_spot, surface_to_camera=surface_to_camera)

    def _detect_tags(self, frame):
        detected_corners, detected_ids, _ = self.tag_detector.detectMarkers(frame)
        cv2.aruco.drawDetectedMarkers(frame, detected_corners, detected_ids)
 
        tags = {}
        if detected_ids is not None:
            for corners, id in zip(detected_corners, detected_ids.flatten()):
                tags.setdefault(id, []).append(corners.reshape(4, 2))

        return tags

    def _get_surface_to_camera(self, tags):
        expected_ids = list(SURFACE_TAG_POINTS.keys())
        if set(expected_ids) - set(tags.keys()):
            return None

        success, rotation_vector, translation_vector = cv2.solvePnP(
            np.concatenate([SURFACE_TAG_POINTS[id] for id in expected_ids]),
            np.concatenate([tags[id][0] for id in expected_ids]),
            CAMERA_MATRIX,
            DIST_COEFFICIENTS
        )

        if not success:
            raise RuntimeError("Failed to solve PnP.")

        return rotation_vector_to_pose(rotation_vector, translation_vector)

    def _get_pieces(self, frame, tags, surface_to_camera):
        pieces = []
        for id, piece_type in PIECE_IDS.items():
            for corners in tags.get(id, []):
                pose = self._get_piece_pose(corners, surface_to_camera)
                pieces.append(DetectedPiece(type=piece_type, square=surface_to_square(pose), pose=pose))
                self._draw_piece_center(frame, surface_to_camera, pose)

        return pieces

    def _draw_piece_center(self, frame, surface_to_camera, pose):
        center_pixel = camera_to_pixel(surface_to_camera @ pose)
        tip_pixel = camera_to_pixel(surface_to_camera @ move_along_axis(pose, 0, 16))

        cv2.circle(frame, center_pixel, 5, (0, 0, 255), -1)
        cv2.arrowedLine(frame, center_pixel, tip_pixel, (0, 0, 255), 2, tipLength=0.3)

    def _get_piece_pose(self, corners, surface_to_camera):
        centre = pixel_to_surface(corners.mean(axis=0), surface_to_camera, PIECE_HEIGHT)
        start = pixel_to_surface(corners[0], surface_to_camera, PIECE_HEIGHT)
        end = pixel_to_surface(corners[1], surface_to_camera, PIECE_HEIGHT)

        heading = end - start

        return roll_pitch_yaw_to_pose(centre, yaw=np.arctan2(heading[1], heading[0]))

    def _get_parking_spot(self, frame, pieces, surface_to_camera):
        half_board = 4 * SQUARE_SIZE
        half_tag = SURFACE_TAG_SIZE / 2
        clearance = PIECE_DIAMETER / 2 + PARKING_TAG_MARGIN
        half_span_x, half_span_y = SURFACE_TAG_SPAN / 2

        inner_x = half_board + PIECE_DIAMETER / 2 + PARKING_BOARD_MARGIN
        outer_x = half_span_x + half_tag - clearance
        limit_y = half_span_y - half_tag - clearance

        parked = [piece.pose[:2, 3] for piece in pieces if piece.square is None]
        min_distance = PIECE_DIAMETER + PARKING_PIECE_CLEARANCE

        for x in np.arange(inner_x, outer_x + 1, PARKING_RESOLUTION):
            for side in (-1, 1):
                for y in np.arange(-limit_y, limit_y + 1, PARKING_RESOLUTION):
                    point = np.array([side * x, y])
                    if all(np.linalg.norm(point - other) >= min_distance for other in parked):
                        self._draw_parking_spot(frame, surface_to_camera, point)
                        return make_pose(np.eye(3), [point[0], point[1], 0.0])

        return None

    def _draw_parking_spot(self, frame, surface_to_camera, point):
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        outline = [
            camera_to_pixel(surface_to_camera @ make_pose(np.eye(3), [
                point[0] + PIECE_DIAMETER / 2 * np.cos(angle),
                point[1] + PIECE_DIAMETER / 2 * np.sin(angle),
                0.0,
            ]))
            for angle in angles
        ]

        cv2.polylines(frame, [np.array(outline)], True, (0, 255, 0), 1)

    def quit(self):
        self.capture.release()

