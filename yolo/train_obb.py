from pathlib import Path
from ultralytics import YOLO

YOLO_DIR = Path(__file__).parent
MODEL_PATH = YOLO_DIR / "yolo26n-obb.pt"
DATASET_CONFIG = YOLO_DIR / "dataset" / "data.yaml"
RUN_DIR = YOLO_DIR / "runs" / "obb" / "icon"

model = YOLO(MODEL_PATH)

model.train(
    data=str(DATASET_CONFIG),
    project=str(RUN_DIR.parent),
    name=RUN_DIR.name,
    epochs=200,
    imgsz=1280,
    batch=4,
    workers=2,
    cache=True,
    patience=50,
    device="mps",
    degrees=180.0,
    scale=0.3,
)