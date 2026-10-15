import cv2, json, shutil
import numpy as np
from pathlib import Path

YOLO_DIR = Path(__file__).parent
DATASET_DIR = YOLO_DIR / "dataset"
PHOTO_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"
SPLITS = ("train", "val")


def find_labelme_files():
    return {split: sorted((PHOTO_DIR / split).glob("*.json")) for split in SPLITS}


def shape_to_label_line(shape, photo_width, photo_height):
    box_corners = cv2.boxPoints(cv2.minAreaRect(np.array(shape["points"], dtype=np.float32)))
    normalised_corners = " ".join(f"{x / photo_width:.6f} {y / photo_height:.6f}" for x, y in box_corners)
    return f"0 {normalised_corners}"


def write_label_file(labelme_file, split_label_dir):
    labelme_data = json.loads(labelme_file.read_text())
    photo_width, photo_height = labelme_data["imageWidth"], labelme_data["imageHeight"]
    label_lines = [
        shape_to_label_line(shape, photo_width, photo_height)
        for shape in labelme_data["shapes"]
        if shape["shape_type"] == "polygon"
    ]
    (split_label_dir / f"{labelme_file.stem}.txt").write_text("\n".join(label_lines))


if __name__ == "__main__":
    labelme_files_by_split = find_labelme_files()

    if LABEL_DIR.exists():
        shutil.rmtree(LABEL_DIR)
    for split, labelme_files in labelme_files_by_split.items():
        split_label_dir = LABEL_DIR / split
        split_label_dir.mkdir(parents=True, exist_ok=True)
        for labelme_file in labelme_files:
            write_label_file(labelme_file, split_label_dir)
