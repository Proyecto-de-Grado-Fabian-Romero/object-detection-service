import os

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from app.usecases.preprocess_equirect import preprocess_image
from app.usecases.run_object_detection import run_detection_on_folder

detect_blueprint = Blueprint("detect", __name__)

UPLOAD_FOLDER = "temp_uploads"
PREPROCESS_OUTPUT = "output_views"


@detect_blueprint.route("/", methods=["POST"])
def detect_from_360_images():
    """
    Detect objects in 360° images

    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: files
        type: file
        required: true
        description: One or more 360° .jpg or .png images
    responses:
      200:
        description: Detections for each 360° image
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PREPROCESS_OUTPUT, exist_ok=True)

    results = []

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Step 1: Preprocess
        folder = preprocess_image(filepath, PREPROCESS_OUTPUT)

        # Step 2: Detect objects
        detections = run_detection_on_folder(folder)

        results.append(
            {
                "original_file": filename,
                "preprocess_folder": folder,
                "views_detected": detections,
            }
        )

    return jsonify(results)
