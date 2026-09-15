import glob, os, shutil
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

PROMPT = ["custom 3d printed white chess piece with dark blue icon, not just the icon, whole white piece"]

shutil.rmtree("dataset/labels", ignore_errors=True)
shutil.rmtree("runs/segment/predict", ignore_errors=True)

overrides = {
    "conf": 0.25,
    "task": "segment",
    "mode": "predict",
    "model": "sam3.pt",
    "device": "mps"
}
predictor = SAM3SemanticPredictor(overrides=overrides)

for split in ["train", "val"]:
    image_dir = f"dataset/images/{split}"
    label_dir = f"dataset/labels/{split}"
    os.makedirs(label_dir, exist_ok=True)

    for image_path in glob.glob(f"{image_dir}/*.jpg"):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")

        predictor.set_image(image_path)
        results = predictor(text=PROMPT)

        lines = []

        if (len(results) > 0 and results[0].masks is not None and len(results[0].masks.xyn) > 0):
            confidences = results[0].boxes.conf.tolist()
            best_index = confidences.index(max(confidences))
            mask_data = results[0].masks.xyn[best_index]

            if isinstance(mask_data, list):
                outer_polygon = max(mask_data, key=len)
            elif isinstance(mask_data, np.ndarray) and mask_data.ndim == 3:
                outer_polygon = max(mask_data, key=len)
            else:
                outer_polygon = mask_data

            if len(outer_polygon) > 0:
                coords_str = " ".join([f"{coord:.6f}" for coord in outer_polygon.flatten()])
                lines.append(f"0 {coords_str}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))