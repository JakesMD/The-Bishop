import chess
import cv2
import numpy as np
from src.detected_state import DetectedState
from src.constants import PIECE_IDS, SQUARE_SIZE, TAG_SIZE, TAG_SPACING, CAMERA_MATRIX, CAMERA_TO_FLANGE, DIST_COEFFICIENTS
from ultralytics import YOLO


class Vision:
    def __init__(self):
        self.model = YOLO("yolo/runs/segment/train/weights/best.pt")
        self.capture = cv2.VideoCapture(0)
        self.tag_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5),
            cv2.aruco.DetectorParameters()
        )

        self.tag_points = []
        half_tag = TAG_SIZE / 2
        half_spacing = TAG_SPACING / 2
        for (sign_x, sign_y) in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
            center_x, center_y = sign_x * half_spacing, sign_y * half_spacing
            self.tag_points.append(np.array([
                [center_x - half_tag, center_y + half_tag, 0.0],
                [center_x + half_tag, center_y + half_tag, 0.0],
                [center_x + half_tag, center_y - half_tag, 0.0],
                [center_x - half_tag, center_y - half_tag, 0.0],
            ]))

    def detect_state(self, board, flange_to_base):
        successful, frame = self.capture.read()
        if not successful:
            raise Exception("Couldn't capture image")

        results = self.model(frame, conf=0.5, verbose=False)
        plotted_frame = results[0].plot(font_size=8, line_width=1)

        board_to_camera = self._get_board_to_camera(plotted_frame)

        board = board.copy()
        board.clear_board()
        piece_positions = {}
        for piece_type, centroids in self._get_piece_centroids(plotted_frame, results[0]).items():
            for centroid in centroids:
                if board_to_camera is not None:
                    base_point, board_point = self._get_pixel_coordinates(centroid, flange_to_base, board_to_camera)
                    square = self._get_square(board_point)
                    if square:
                        board.set_piece_at(square, piece_type)

        return DetectedState(frame=plotted_frame, board=board)

    def _get_piece_centroids(self, frame, results):
        if results.masks is None:
            return {}

        centroids = {}
        class_ids = results.boxes.cls.int().tolist()
        for class_id, contour_points in zip(class_ids, results.masks.xy):
            M = cv2.moments(contour_points)
            area, sum_x, sum_y = M["m00"], M["m10"], M["m01"]
            centroid = np.array([sum_x / area, sum_y / area])
            centroids.setdefault(PIECE_IDS[class_id], []).append(centroid)

            cv2.circle(
                img=frame,
                center=[round(centroid[0]), round(centroid[1])],
                radius=6,
                color=(0, 255, 0),
                thickness=-1
            )

        return centroids

    def _get_board_to_camera(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_corners, detected_ids, _ = self.tag_detector.detectMarkers(gray)

        if detected_ids is None:
            return None

        detected_tag_points = [None, None, None, None]
        for corners, id in zip(detected_corners, detected_ids.flatten()):
            if id < 4:
                detected_tag_points[id] = corners.reshape(4, 2)

        if any(tag is None for tag in detected_tag_points):
            return None

        success, rotation_vector, translation_vector = cv2.solvePnP(
            np.concatenate(self.tag_points),
            np.concatenate(detected_tag_points),
            CAMERA_MATRIX,
            None # DIST_COEFFICIENTS
        )

        if not success:
            return Exception("Failed to solve PnP.")

        board_to_camera = np.eye(4)
        board_to_camera[:3, :3], _ = cv2.Rodrigues(rotation_vector)
        board_to_camera[:3, 3] = translation_vector.flatten()

        cv2.aruco.drawDetectedMarkers(frame, detected_corners, detected_ids)
        tag_centers = np.array([corners.mean(axis=0) for corners in detected_tag_points], dtype=np.int32)
        cv2.polylines(frame, [tag_centers], isClosed=True, color=(0, 0, 255), thickness=3)

        return board_to_camera

    def _get_pixel_coordinates(self, pixel, flange_to_base, board_to_camera):
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

        return base_point, board_point

    def _get_square(self, board_point):
        half_board = 4 * SQUARE_SIZE
        file_index = 7 - int((board_point[0] + half_board) // SQUARE_SIZE)
        rank_index = 7 - int((board_point[1] + half_board) // SQUARE_SIZE)

        if not (0 <= file_index < 8 and 0 <= rank_index < 8):
            return None

        return chess.square(file_index, rank_index)

    def quit(self):
        self.capture.release()

