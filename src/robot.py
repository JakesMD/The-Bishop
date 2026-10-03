import franky as fk
import numpy as np
from scipy.spatial.transform import Rotation
from config import *
from src.transforms import *

class Robot:
    def __init__(self, wait_for_retry):
        self._wait_for_retry = wait_for_retry
  #      self._arm = fk.Robot("172.16.0.2")
  #      self._gripper = fk.Gripper("172.16.0.2")

  #      self._arm.recover_from_errors()
  #      self._arm.relative_dynamics_factor = 0.1
  #      # self._gripper.homing()

    def get_pose(self):
        return np.eye(4)
        self._arm.recover_from_errors()
        pose = self._arm.current_cartesian_state.pose.end_effector_pose.matrix
        return fix_rotation_error(pose * 1000)
        

    def move_to(self, pose):
        while True:
            try:
               # raise Exception()
                break
                self._arm.stop()
                self._arm.recover_from_errors()
                quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
                motion = fk.CartesianMotion(fk.Affine(pose[:3, 3] * 0.001, quat), reference_type=fk.ReferenceType.Absolute)
                self._arm.move(motion)
                break
            except Exception as e:
                self._wait_for_retry(e)

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
