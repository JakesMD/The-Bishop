import cv2, json, random, shutil, sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO

YOLO_DIR = Path(__file__).parent
sys.path.insert(0, str(YOLO_DIR.parent))
from config import *

MODEL_PATH = YOLO_DIR / "yolo26n-cls.pt"
PHOTO_DIR = YOLO_DIR / "dataset" / "images"
CROP_DATASET_DIR = YOLO_DIR / "dataset" / "cls"
RUN_DIR = YOLO_DIR / "runs" / "classify" / "train"
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720
JITTERED_COPIES = 6


def crop_icon(image, centre_x, centre_y, side_length, angle_degrees):
    transform = cv2.getRotationMatrix2D((centre_x, centre_y), angle_degrees, ICON_CROP_SIZE_PX / side_length)
    transform[:, 2] += ICON_CROP_SIZE_PX / 2 - np.array([centre_x, centre_y])
    return cv2.warpAffine(image, transform, (ICON_CROP_SIZE_PX, ICON_CROP_SIZE_PX), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def save_crops(crop, class_dir, name_prefix, copy_index):
    for quarter_turns in range(4):
        crop_file = class_dir / f"{name_prefix}_{copy_index}_{quarter_turns}.jpg"
        cv2.imwrite(str(crop_file), np.rot90(crop, quarter_turns), [cv2.IMWRITE_JPEG_QUALITY, 95])


def make_crop(camera_image, centre_x, centre_y, crop_side_length, icon_angle, copy_index):
    if copy_index == 0:
        return crop_icon(camera_image, centre_x, centre_y, crop_side_length, icon_angle)
    return crop_icon(
        camera_image,
        centre_x + random.uniform(-2, 2),
        centre_y + random.uniform(-2, 2),
        crop_side_length * random.uniform(0.9, 1.15),
        icon_angle + random.uniform(-7, 7),
    )


def find_icon_box(shape, camera_scale):
    icon_outline = (np.array(shape["points"]) * camera_scale).astype(np.float32)
    (centre_x, centre_y), (icon_width, icon_height), icon_angle = cv2.minAreaRect(icon_outline)
    return centre_x, centre_y, max(icon_width, icon_height) * ICON_CROP_MARGIN_RATIO, icon_angle


def build_icon_crops(split, camera_image, shape, camera_scale, name_prefix):
    class_dir = CROP_DATASET_DIR / split / shape["label"]
    class_dir.mkdir(parents=True, exist_ok=True)
    centre_x, centre_y, crop_side_length, icon_angle = find_icon_box(shape, camera_scale)
    for copy_index in range(JITTERED_COPIES if split == "train" else 1):
        crop = make_crop(camera_image, centre_x, centre_y, crop_side_length, icon_angle, copy_index)
        save_crops(crop, class_dir, name_prefix, copy_index)


def build_photo_crops(split, labelme_file):
    labelme_data = json.loads(labelme_file.read_text())
    camera_scale = (CAMERA_WIDTH / labelme_data["imageWidth"], CAMERA_HEIGHT / labelme_data["imageHeight"])
    photo = cv2.imread(str(PHOTO_DIR / split / f"{labelme_file.stem}.jpg"))
    camera_image = cv2.resize(photo, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)
    polygons = [shape for shape in labelme_data["shapes"] if shape["shape_type"] == "polygon"]
    for icon_index, shape in enumerate(polygons):
        build_icon_crops(split, camera_image, shape, camera_scale, f"{labelme_file.stem}_{icon_index}")


if __name__ == "__main__":
    random.seed(0)
    shutil.rmtree(CROP_DATASET_DIR, ignore_errors=True)
    for split in ("train", "val"):
        for labelme_file in sorted((PHOTO_DIR / split).glob("*.json")):
            build_photo_crops(split, labelme_file)

    model = YOLO(MODEL_PATH)
    model.train(
        data=str(CROP_DATASET_DIR),
        project=str(RUN_DIR.parent),
        name=RUN_DIR.name,
        epochs=25,
        imgsz=ICON_CROP_SIZE_PX,
        batch=64,
        workers=2,
        device="mps",
        fliplr=0.0,
        scale=0.15,
        erasing=0.0,
        auto_augment=None,
    )
