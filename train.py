from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

model.train(
    data="dataset/data.yml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",

    degrees=180.0,
    translate=0.1,
    scale=0.3,
    shear=2.0,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.0,

    hsv_h=0.015,
    hsv_s=0.1,
    hsv_v=0.6,

    mosaic=0.5,
    copy_paste=0.0
)