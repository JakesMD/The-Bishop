import cv2
import numpy as np
from ultralytics.engine.results import Results


def get_piece_centroids(frame: np.ndarray, results: Results) -> list[np.ndarray]:
    if results.masks is None:
        return []

    centroids = []
    for contour_points in results.masks.xy:
        M = cv2.moments(contour_points)
        area = M["m00"]
        sum_x = M["m10"]
        sum_y = M["m01"]

        centroid = np.array([sum_x / area, sum_y / area])

        cv2.circle(
            img=frame,
            center=[round(centroid[0]), round(centroid[1])],
            radius=6,
            color=(0, 255, 0),
            thickness=-1
        )

        centroids.append(centroid)

    return centroids
