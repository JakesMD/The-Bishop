import cv2
import numpy as np
import chess.svg
import cairosvg
from config import *
from src.transforms import *
from src.types import *


class Quit(Exception):
    pass

_quit_cmd = Command("q", "quit")

class Canvas:
    def __init__(self):
        self._width = 1920
        self._panel_height = 480
        self._camera_height = 1080
        self._margin = 40

        self._font_scale = 0.9
        self._line_height = 50
        self._board_size = self._panel_height - self._line_height

        self.window_name = "The Bishop"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self._width, self._panel_height + self._camera_height)

        self.board_image = None
        self.detected_image = None
        self.frame = None
        self.robot_configuration = None
        self.camera_to_base = None
        self.text = ""
        self.commands = []
        self._last_key = -1

    def set_text(self, text):
        self.text = text

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

    def set_commands(self, commands=()):
        self.commands = list(commands)

    def draw(self):
        camera_view = self.frame if self.frame is not None else np.zeros((self._camera_height, self._width, 3), np.uint8)

        if self.robot_configuration is not None:
            camera_view = self._draw_end_effector(camera_view.copy(), self.robot_configuration, np.linalg.inv(self.camera_to_base))

     #   camera_view = cv2.rotate(camera_view, cv2.ROTATE_180)
        height, width, _ = camera_view.shape
        camera_view = cv2.resize(camera_view, (self._width, int(height * self._width / width)))

        cv2.imshow(self.window_name, np.vstack((self._draw_panel(), camera_view)))

        self._last_key = cv2.waitKey(1) & 0xFF
        if self._last_key == ord(_quit_cmd.key):
            raise Quit()

    def read_command(self):
        return self._consume(self._last_key)

    def wait_for_command(self):
        while True:
            command = self._consume(cv2.waitKey(0) & 0xFF)
            if command is not None:
                return command

    def _consume(self, key):
        if key == ord(_quit_cmd.key):
            raise Quit()

        for command in self.commands:
            if key == ord(command.key):
                self.commands = []
                return command

    def _draw_panel(self):
        panel = np.zeros((self._panel_height, self._width, 3), np.uint8)

        right = self._width
        for image, caption in [(self.detected_image, "Detected"), (self.board_image, "Expected")]:
            if image is None:
                continue

            right -= self._board_size
            panel[self._line_height:, right:right + self._board_size] = image
            self._draw_caption(panel, caption, right)

        y = self._margin + self._line_height
        cv2.putText(panel, self.text, (self._margin, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        commands = [*self.commands, _quit_cmd]
        key_width = max(
            cv2.getTextSize(command.label, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, 2)[0][0]
            for command in commands
        )

        y += self._line_height // 2
        for command in commands:
            y += self._line_height
            cv2.putText(panel, command.label, (self._margin, y), cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, (110, 110, 110), 2)
            cv2.putText(
                panel, command.description,
                (self._margin + key_width + self._margin, y),
                cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, (110, 110, 110), 2,
            )

        return panel

    def _draw_caption(self, panel, caption, left):
        text_width = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, 2)[0][0]
        origin = (left + (self._board_size - text_width) // 2, self._line_height - self._margin // 4)
        cv2.putText(panel, caption, origin, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, (180, 180, 180), 2)

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