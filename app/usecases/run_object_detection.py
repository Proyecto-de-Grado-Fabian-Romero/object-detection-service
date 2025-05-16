import json
import os
from typing import Dict, List

import cv2

from app.adapters.object_detection.yolo_inference import load_model, predict


def run_detection_on_folder(
    folder_path: str,
    model_path: str = "best.pt",
    conf: float = 0.5,
    classes: List[int] = [],
) -> List[Dict]:
    model = load_model(model_path)

    metadata_file = os.path.join(folder_path, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    results = []

    for view in metadata:
        image_path = os.path.join(folder_path, view["filename"])
        img = cv2.imread(image_path)
        detections = predict(model, img, classes=classes, conf=conf)[0]

        boxes = []
        for box in detections.boxes:
            boxes.append(
                {
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "xyxy": [float(v) for v in box.xyxy[0]],
                }
            )

        results.append(
            {
                "filename": view["filename"],
                "yaw": view["yaw"],
                "pitch": view["pitch"],
                "fov": view["fov"],
                "detections": boxes,
            }
        )

    return results
