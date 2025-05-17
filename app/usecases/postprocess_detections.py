from typing import Dict, List, Union

import cv2
import numpy as np

from app.adapters.image_processing.coordinate_mapper import (
    perspective_bbox_to_equirectangular,
)
from app.adapters.tracking.deep_sort_tracking import DeepSortTracker  # tu clase
from app.entities.class_names import CLASS_ID_TO_NAME
from app.typing.class_stats import ClassStats


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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


def postprocess_detections_with_tracking(
    detections: List[Dict], original_360_img_path: str, iou_threshold=0.05
) -> Dict[str, ClassStats]:
    """
    Aplica NMS global y luego DeepSORT para tracking.
    Devuelve conteo de objetos únicos con nombres en español.
    """

    img_360 = cv2.imread(original_360_img_path)
    if img_360 is None:
        raise FileNotFoundError(
            f"Original 360 image not found: {original_360_img_path}"
        )

    h_eq, w_eq = img_360.shape[:2]
    tracker = DeepSortTracker()

    all_boxes = []
    all_scores = []
    all_class_ids = []

    # For each view, map bboxes to equirectangular coordinates
    for view in detections:
        yaw = view["yaw"]
        pitch = view["pitch"]
        fov = view["fov"]
        w_out, h_out = 512, 512  # tamaño salida asumido

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

    # NMS global by class
    filtered_boxes = []
    filtered_scores = []
    filtered_class_ids = []

    for class_id in np.unique(class_ids_np):
        inds = np.where(class_ids_np == class_id)[0]
        boxes_class = boxes_np[inds]
        scores_class = scores_np[inds]

        keep = non_max_suppression(boxes_class, scores_class, iou_threshold)

        filtered_boxes.extend(boxes_class[keep])
        filtered_scores.extend(scores_class[keep])
        filtered_class_ids.extend([class_id] * len(keep))

    # Prepare detections for DeepSORT
    detections_for_tracking = []
    for bbox, score, class_id in zip(
        filtered_boxes, filtered_scores, filtered_class_ids
    ):
        detections_for_tracking.append([bbox, score, class_id])

    print(detections_for_tracking)

    detections_for_tracking = [
        [det[0], det[1], det[2]] for det in detections_for_tracking
    ]

    # Update tracker with equirectangular image and filtered detections
    tracked_objects = tracker.update(img_360, detections_for_tracking)

    # Count unique objects by track_id and class
    objects_by_id = {}
    for obj in tracked_objects:
        track_id = obj["track_id"]
        if track_id not in objects_by_id:
            objects_by_id[track_id] = {"class_id": obj["class_id"], "count": 0}
        objects_by_id[track_id]["count"] += 1

    # Group by name, id and count objects
    result: Dict[str, ClassStats] = {}
    for obj in objects_by_id.values():
        class_id = obj["class_id"]
        class_id_str = str(class_id)
        name = CLASS_ID_TO_NAME.get(class_id)
        if name:
            if class_id_str not in result:
                result[class_id_str] = {"name": name, "count": 1}
            else:
                result[class_id_str]["count"] += 1

    return result
