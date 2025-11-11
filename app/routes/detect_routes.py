import os
from io import BytesIO

import requests
from flask import Blueprint, jsonify, request
from PIL import Image
from werkzeug.utils import secure_filename

from app.usecases.postprocess_detections import postprocess_detections_with_tracking
from app.usecases.preprocess_equirect import preprocess_image
from app.usecases.run_object_detection import run_detection_on_folder

detect_blueprint = Blueprint("detect", __name__)

UPLOAD_FOLDER = "temp_uploads"
PREPROCESS_OUTPUT = "output_views"


@detect_blueprint.route("/", methods=["POST"])
def detect():
    """
    Detect objects in 360º equirectangular images from URLs.

    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: image_urls
        description: List of URLs of the 360º .jpg or .png images to detect objects in
        required: true
        type: array
        items:
          type: string
          format: url
    responses:
      200:
        description: Object counts detected in the images
        schema:
            type: object
            additionalProperties:
                type: object
                properties:
                name:
                    type: string
                count:
                    type: integer
                required:
                - name
                - count
      400:
        description: Error due to invalid input or download failure
    """
    image_urls = request.json
    if not image_urls:
        return jsonify({"error": "No image URLs provided"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PREPROCESS_OUTPUT, exist_ok=True)

    aggregated_objects = {}

    for url in image_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            filename = secure_filename(url.split("/")[-1])
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            image.save(filepath)

            # Process
            folder = preprocess_image(filepath, PREPROCESS_OUTPUT)
            detections = run_detection_on_folder(folder)
            objects_count = postprocess_detections_with_tracking(detections, filepath)

            # Count objects
            for class_id, data in objects_count.items():
                if class_id not in aggregated_objects:
                    aggregated_objects[class_id] = {"name": data["name"], "count": 0}
                aggregated_objects[class_id]["count"] += data["count"]

            # Delete images
            os.remove(filepath)
            for preprocessed_img in os.listdir(folder):
                os.remove(os.path.join(folder, preprocessed_img))
            os.rmdir(folder)

        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Failed to download image: {str(e)}"}), 400

    return jsonify(aggregated_objects)
