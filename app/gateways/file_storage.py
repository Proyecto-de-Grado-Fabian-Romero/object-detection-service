import os
import json
from typing import List, Tuple

import cv2
import numpy as np
from app.entities.view_metadata import ViewMetadata

def save_views(output_dir: str, views: List[Tuple[np.ndarray, ViewMetadata]]):
    os.makedirs(output_dir, exist_ok=True)

    metadata_list = []
    for idx, (img, meta) in enumerate(views):
        filename = f"view_{idx:03}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)

        meta.filename = filename
        metadata_list.append(meta.__dict__)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata_list, f, indent=2)
