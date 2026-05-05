import os
import uuid

import cv2

from app.adapters.image_processing.perspective_converter import convert_to_perspective
from app.entities.view_metadata import ViewMetadata
from app.gateways.file_storage import save_views


def preprocess_image(image_path: str, output_base_dir: str) -> str:
    img = cv2.imread(image_path)

    # 8 yaws at 45° intervals give overlapping views (50% overlap with 90° FOV)
    # so objects near seams between views are always captured by at least one view
    yaws = [0, 45, 90, 135, 180, 225, 270, 315]
    pitches = [45, 0, -45]  # Up, horizontal, down
    fov = 90
    output_size = (512, 512)

    views = []
    for pitch in pitches:
        for yaw in yaws:
            persp = convert_to_perspective(img, yaw, pitch, fov, output_size)
            meta = ViewMetadata(filename="", yaw=yaw, pitch=pitch, fov=fov)
            views.append((persp, meta))

    folder_id = str(uuid.uuid4())
    output_dir = os.path.join(output_base_dir, folder_id)
    save_views(output_dir, views)

    return output_dir
