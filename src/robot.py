#import franky as fk
import numpy as np
from scipy.spatial.transform import Rotation
from src.constants import *
from src.transforms import *
import time


class Robot:
    def __init__(self):
        return
        self._arm = fk.Robot("172.16.0.2")
        self._gripper = fk.Gripper("172.16.0.2")

        self._arm.recover_from_errors()
        self._arm.relative_dynamics_factor = 0.1
        self._gripper.homing()

    def get_pose(self):
       # self._arm.recover_from_errors()
        pose = np.eye(4) # self._arm.current_cartesian_state.end_effector_pose.matrix
        return fix_rotation_error(pose * 1000)

    def move_to(self, pose):
        time.sleep(1)
        return
        self._arm.recover_from_errors()
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        motion = fk.CartesianMotion(fk.Affine(pose[:3, 3] * 0.001, quat), reference_type=fk.ReferenceType.absolute)
        self._arm.move(motion)

    def close_gripper(self):
        return
        self._arm.recover_from_errors()
        self._gripper.grasp(0.015, 0.02, 20.0)

    def open_gripper(self):
        return
        self._arm.recover_from_errors()
        self._gripper.move(0.02, 0.02)

    def quit(self):
        pass
