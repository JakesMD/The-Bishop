# The Bishop

A robot arm that plays chess against you on a real board. A camera on the arm
reads the board, the
[instinct-chess-bot](https://github.com/JakesMD/instinct-chess-bot) engine picks
a move, and the arm moves the pieces.

This is my bachelor thesis project. It runs on a Franka Emika arm, but any arm
can be swapped in (see [Using a different robot](#using-a-different-robot)).

## How it works

- **Board pose:** four AprilTags (16h5) at the corners of the printed board give
  the board's position relative to the camera.
- **Piece detection:** each piece has an icon on top. A YOLO oriented bounding
  box model finds the icons, and a YOLO classifier identifies each one (colour
  and piece type).
- **Move checking:** the detected board is compared against the last known
  position, and illegal moves are rejected.
- **Motion:** the planner works out which pieces need to move, picks and places
  them, and puts captured pieces in free spots beside the board.

The Bishop plays white and you play black.

## Hardware

- A robot arm with a gripper (Franka Emika supported out of the box)
- A camera mounted on the arm, calibrated (intrinsics and hand-eye)
- The board, printed on three A4 pages: [left](assets/board-left.png),
  [center](assets/board-center.png), [right](assets/board-right.png)
- 3D-printed pieces from the `.step` files in [assets/](assets/). You need 34
  pieces: a full set plus an extra queen for each side for promotions.

## Setup

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Update these values in [config.py](config.py). Distances are in millimetres.

| Setting                              | What it is                                             |
| ------------------------------------ | ------------------------------------------------------ |
| `CAMERA_INDEX`                       | OpenCV index of the arm camera                         |
| `CAMERA_MATRIX`, `DIST_COEFFICIENTS` | Camera intrinsics from calibration                     |
| `CAMERA_TO_FLANGE`                   | Hand-eye calibration                                   |
| `ROBOT_APPROACH_HEIGHT`              | End effector z height above the board when approaching |
| `ROBOT_GRIP_HEIGHT`                  | End effector z height above the board when gripping    |

Then measure your printed board, because printers rarely print at exactly the
right scale:

| Setting            | What to measure                     |
| ------------------ | ----------------------------------- |
| `SQUARE_SIZE`      | Side length of one square           |
| `SURFACE_TAG_SIZE` | Side length of one AprilTag         |
| `SURFACE_TAG_SPAN` | Distance between tag centres (x, y) |

## Playing

```shell
source .venv/bin/activate
python3 play.py
```

1. Move the arm so the camera sees the whole board, then continue. This viewing
   pose is saved to `viewing_pose.npy`, so next time you can send the arm back
   to it.
2. The arm sets up any pieces that are not in their starting squares.
3. Make your move, then press space to end your turn.
4. The Bishop replies by moving its piece.

The available keys are shown on screen. `r` resets the board and `q` quits.

## Using a different robot

Subclass `Robot` from [src/robot.py](src/robot.py) and implement `_get_pose`,
`_move_to`, `_close_gripper` and `_open_gripper`, using
[src/franka_emika.py](src/franka_emika.py) as a template. Poses are 4×4 matrices
in millimetres. Then swap `FrankaEmika` for your class in [play.py](play.py).

## Retraining the vision models

The trained weights are included in [yolo/runs/](yolo/runs/). To retrain them on
your own photos:

1. Label the piece icons as polygons with
   [labelme](https://github.com/wkentaro/labelme), naming each label
   `<colour>_<piece>` (e.g. `white_knight`, `black_queen`). Put the `.jpg`
   photos and their `.json` files in `yolo/dataset/images/train` and
   `yolo/dataset/images/val`.
2. Convert the labels: `python3 yolo/labelme2yolo.py`
3. Train the detector: `python3 yolo/train_obb.py`
4. Train the classifier: `python3 yolo/train_cls.py`
