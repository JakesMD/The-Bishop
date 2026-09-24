import cv2
import numpy as np
import chess.svg
import cairosvg
from src.constants import *
from src.transforms import *


class Quit(Exception):
    pass

class Reset(Exception):
    pass


class Canvas:
    def __init__(self):
        self._width = 1920
        self._header_height = 480
        self._body_height = 1080
        self._margin = 40

        self._message_scale = 0.9
        self._line_height = 50
        self._board_size = self._header_height - self._line_height

        self.window_name = "The Bishop"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self._width, self._header_height + self._body_height)

        self.board_image = None
        self.detected_image = None
        self.frame = None
        self.robot_configuration = None
        self.camera_to_base = None
        self.title = ""
        self.message = ""
        self.commands = []

    def set_text(self, title, message=""):
        self.title = title
        self.message = message

    def set_board(self, board):
        self.board_image = self._draw_board(board)

    def set_detected_board(self, board):
        self.detected_image = self._draw_board(board)

    def set_frame(self, frame):
        self.frame = frame
        self.robot_configuration = None

    def set_robot_configuration(self, robot_configuration, camera_to_base):
        self.robot_configuration = robot_configuration
        self.camera_to_base = camera_to_base

    def display(self):
        body = self.frame if self.frame is not None else np.zeros((self._body_height, self._width, 3), np.uint8)

        if self.robot_configuration is not None:
            body = self._draw_end_effector(body.copy(), self.robot_configuration, np.linalg.inv(self.camera_to_base))

        height, width, _ = body.shape
        body = cv2.resize(body, (self._width, int(height * self._width / width)))

        cv2.imshow(self.window_name, np.vstack((self._draw_header(), body)))
        self._read_key()

    def wait_for_space_press(self, space="continue"):
        self.commands = [("SPACE", space)]
        self.display()

        while self._read_key() != ord(" "):
            pass

        self.commands = []

    def wait(self):
       while True:
            self._read_key()

    def _read_key(self):
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            raise Quit()
        if key == ord("r"):
            raise Reset()

        return key

    def _draw_header(self):
        header = np.zeros((self._header_height, self._width, 3), np.uint8)

        right = self._width
        for image, caption in [(self.detected_image, "Detected"), (self.board_image, "Expected")]:
            if image is None:
                continue

            right -= self._board_size
            header[self._line_height:, right:right + self._board_size] = image
            self._draw_caption(header, caption, right)

        y = self._margin + self._line_height
        cv2.putText(header, self.title, (self._margin, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

        if self.message:
            y += self._line_height // 2 + self._line_height
            cv2.putText(header, self.message, (self._margin, y), cv2.FONT_HERSHEY_SIMPLEX, self._message_scale, (180, 180, 180), 2)

        commands = self.commands + [("R", "reset"), ("Q", "quit")]
        key_width = max(
            cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, self._message_scale, 2)[0][0]
            for key, _ in commands
        )

        y += self._line_height // 2
        for key, description in commands:
            y += self._line_height
            cv2.putText(header, key, (self._margin, y), cv2.FONT_HERSHEY_SIMPLEX, self._message_scale, (110, 110, 110), 2)
            cv2.putText(
                header, description,
                (self._margin + key_width + self._margin, y),
                cv2.FONT_HERSHEY_SIMPLEX, self._message_scale, (110, 110, 110), 2,
            )

        return header

    def _draw_caption(self, header, caption, left):
        text_width = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, self._message_scale, 2)[0][0]
        origin = (left + (self._board_size - text_width) // 2, self._line_height - self._margin // 4)
        cv2.putText(header, caption, origin, cv2.FONT_HERSHEY_SIMPLEX, self._message_scale, (180, 180, 180), 2)

    def _draw_board(self, board: chess.Board):
        check_square = board.king(board.turn) if board.is_check() else None
        board_svg = chess.svg.board(board=board, size=self._board_size, flipped=True, check=check_square)
        png_bytes = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'))
        return cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _draw_end_effector(self, frame, robot_configuration, base_to_camera):
        origin_pixel = camera_to_pixel(base_to_camera @ robot_configuration.pose)

        for axis, colour in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            tip = base_to_camera @ move_along_axis(robot_configuration.pose, axis, 40.0)
            cv2.arrowedLine(frame, origin_pixel, camera_to_pixel(tip), colour, 2, tipLength=0.2)

        if robot_configuration.is_gripper_closed:
            cv2.circle(frame, origin_pixel, 6, (0, 255, 255), -1)

        return frame

    def quit(self):
        cv2.destroyAllWindows()