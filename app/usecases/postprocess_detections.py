import os
import cv2
import numpy as np
from typing import List, Dict
from app.adapters.image_processing.coordinate_mapper import perspective_bbox_to_equirectangular
from app.entities.class_names_es import CLASS_ID_TO_NAME_ES

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Simple NMS implementation for bounding boxes.
    boxes: np.array shape (N, 4) in format [xmin, ymin, xmax, ymax]
    scores: np.array shape (N,)
    """
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def postprocess_detections(detections: List[Dict], original_360_img_path: str, iou_threshold=0.5) -> Dict[str, int]:
    """
    detections: Receives detections in format:
    [
      {
        "filename": "...",
        "yaw": ...,
        "pitch": ...,
        "fov": ...,
        "detections": [
          {"class_id": int, "confidence": float, "xyxy": [xmin,ymin,xmax,ymax]},
          ...
        ]
      },
      ...
    ]
    original_360_img_path: path to the original equirectangular image to get size

    Returns count of objects by class with names in Spanish.
    """

    # Upload 360 image to get resolution
    img_360 = cv2.imread(original_360_img_path)
    if img_360 is None:
        raise FileNotFoundError(f"Original 360 image not found: {original_360_img_path}")

    h_eq, w_eq = img_360.shape[:2]

    all_boxes = []
    all_scores = []
    all_class_ids = []

    # For each view, map bboxes to equirectangular coordinates
    for view in detections:
        yaw = view["yaw"]
        pitch = view["pitch"]
        fov = view["fov"]

        # Output size (we assume 512x512, change if it is other)
        w_out = 512
        h_out = 512

        for det in view["detections"]:
            bbox_persp = det["xyxy"]
            bbox_eq = perspective_bbox_to_equirectangular(
                bbox_persp, w_out, h_out, yaw, pitch, fov, w_eq, h_eq
            )

            all_boxes.append(bbox_eq)
            all_scores.append(det["confidence"])
            all_class_ids.append(det["class_id"])

    if not all_boxes:
        return {}

    boxes_np = np.array(all_boxes)
    scores_np = np.array(all_scores)
    class_ids_np = np.array(all_class_ids)

    final_class_counts = {}

    # Global NMS by class in equirectangular
    for class_id in np.unique(class_ids_np):
        inds = np.where(class_ids_np == class_id)[0]
        boxes_class = boxes_np[inds]
        scores_class = scores_np[inds]

        keep = non_max_suppression(boxes_class, scores_class, iou_threshold)

        final_class_counts[class_id] = len(keep)

    # Translates into Spanish names
    result = {}
    for class_id, count in final_class_counts.items():
        name_es = CLASS_ID_TO_NAME_ES.get(class_id, str(class_id))
        result[name_es] = count

    return result
