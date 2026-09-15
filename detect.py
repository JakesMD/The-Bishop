import cv2
import numpy as np
from ultralytics import YOLO

from board_to_camera import get_board_to_camera
from piece_centroids import get_piece_centroids
from pixel_coordinates import get_pixel_coordinates

flange_to_base = np.array([
    [-0.5685,   0.82168,   -0.04076,  -365.341],
    [0.82176,   0.56951,    0.0192,     95.315],
    [0.03899,  -0.02257,   -0.99898,   509.752],
    [0,         0,          0,           1    ],
])

model = YOLO("runs/segment/train/weights/best.pt")
capture = cv2.VideoCapture(0)

while True:
    successful, frame = capture.read()
    results = model(frame, conf=0.5)
    plotted_frame = results[0].plot()

    board_to_camera = get_board_to_camera(plotted_frame)

    for centroid in get_piece_centroids(plotted_frame, results[0]):
        if board_to_camera is not None:
            base_point, board_point = get_pixel_coordinates(plotted_frame, centroid, flange_to_base, board_to_camera)

    cv2.imshow("Detect", plotted_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()