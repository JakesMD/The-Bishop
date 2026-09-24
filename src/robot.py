import time

import numpy as np
from src.constants import *


class Robot:
    def __init__(self):
        self.pose = np.eye(4)
        self.is_gripper_closed = False

    def get_pose(self):
        return self.pose.copy()

    def move_to(self, pose):
        self.pose = pose.copy()
        time.sleep(1.5)

    def close_gripper(self):
        self.is_gripper_closed = True
        time.sleep(0.5)

    def open_gripper(self):
        self.is_gripper_closed = False
        time.sleep(0.5)

    def quit(self):
        pass
