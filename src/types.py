from collections import namedtuple


class Command(namedtuple("Command", ["key", "description"])):
    @property
    def label(self):
        return "SPACE" if self.key == " " else self.key.upper()


DetectedPiece = namedtuple("DetectedPiece", ["type", "square", "pose"])

DetectedState = namedtuple("DetectedState", ["frame", "board", "pieces", "parking_spot", "surface_to_camera"])

Transfer = namedtuple("Transfer", ["piece", "square"])

RobotConfiguration = namedtuple("RobotConfiguration", ["pose", "is_gripper_closed"])