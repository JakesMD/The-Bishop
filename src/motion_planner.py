import chess
import numpy as np
from src.constants import *
from src.board import *
from src.transforms import *
from src.types import *


class MotionPlanner:
    def __init__(self, robot):
        self.robot = robot
        self.viewing_pose = None
        self.camera_to_base = None
        self.configuration = None

    def set_viewing_pose(self):
        self.viewing_pose = fix_rotation_error(self.robot.get_pose())
        self.camera_to_base = self.viewing_pose @ CAMERA_TO_FLANGE

    def look_at_board(self):
        self.robot.move_to(self.viewing_pose)
        self.configuration = RobotConfiguration(pose=self.viewing_pose, is_gripper_closed=False)

    def plan_transfer(self, state, target_board):
        surplus, wanted = diff_board(state.board, target_board)

        for square in wanted:
            if state.board.piece_at(square) is None:
                piece = self._get_source_piece(state, square, surplus, wanted)
                if piece is not None:
                    return Transfer(piece=piece, square=square)

        for square in wanted:
            if state.board.piece_at(square) is not None:
                return self._park(state, square)

        for square in surplus:
            if square not in wanted:
                return self._park(state, square)

        return None

    def make_transfer(self, state, transfer):
        pick_pose = self._get_pick_pose(transfer.piece.pose)
        place_pose = self._get_place_pose(state, transfer.square)

        yield self.robot.open_gripper()
        yield self._move_to(state, self._approach(pick_pose))
        yield self._move_to(state, pick_pose)
        yield self._close_gripper()

        yield self._move_to(state, self._approach(pick_pose))
        yield self._move_to(state, self._approach(place_pose))
        yield self._move_to(state, place_pose)
        yield self._open_gripper()

        yield self._move_to(state, self._approach(place_pose))

    def _get_source_piece(self, state, square, surplus, wanted):
        movable = [
            piece for piece in state.pieces
            if piece.square is None or piece.square in surplus
        ]

        candidates = [piece for piece in movable if piece.type == wanted[square]]

        if not candidates:
            wanted_types = set(wanted.values())
            candidates = [
                piece for piece in movable
                if piece.square is not None and piece.type not in wanted_types
            ]

        if not candidates:
            return None

        point = square_to_surface(square)[:2]
        return min(candidates, key=lambda piece: (
            piece.square is None,
            np.linalg.norm(piece.pose[:2, 3] - point),
        ))

    def _park(self, state, square):
        return Transfer(piece=self._get_piece_on_square(state.pieces, square), square=None)

    def _get_piece_on_square(self, pieces, square):
        for piece in pieces:
            if piece.square == square:
                return piece

        raise RuntimeError("No piece detected on " + chess.square_name(square) + ".")

    def _get_place_pose(self, state, square):
        position = self._get_target_position(state, square)
        return self._get_pick_pose(roll_pitch_yaw_to_pose(position, yaw=0))

    def _get_pick_pose(self, pose):
        return move_along_axis(pose, 2, ROBOT_GRIP_HEIGHT) @ roll_pitch_yaw_to_pose([0.0, 0.0, 0.0], roll=np.pi)

    def _get_target_position(self, state, square):
        if square is None:
            if state.parking_spot is None:
                raise RuntimeError("Nowhere left to put a piece.")
            return state.parking_spot[:, 3]

        return square_to_surface(square)

    def _approach(self, pose):
        return move_along_axis(pose, 2, -ROBOT_APPROACH_HEIGHT)

    def _move_to(self, state, pose):
        base_pose = self.camera_to_base @ state.surface_to_camera @ pose
        self.robot.move_to(base_pose)
        self.configuration = self.configuration._replace(pose=base_pose)
        return self.configuration

    def _close_gripper(self):
        self.robot.close_gripper()
        self.configuration = self.configuration._replace(is_gripper_closed=True)
        return self.configuration

    def _open_gripper(self):
        self.robot.open_gripper()
        self.configuration = self.configuration._replace(is_gripper_closed=False)
        return self.configuration
