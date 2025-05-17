from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
from app.usecases.preprocess_equirect import preprocess_image
from app.usecases.run_object_detection import run_detection_on_folder
from app.usecases.postprocess_detections import postprocess_detections

detect_blueprint = Blueprint("detect", __name__)

UPLOAD_FOLDER = "temp_uploads"
PREPROCESS_OUTPUT = "output_views"
MODEL_PATH = "models/yolo11x.pt"

@detect_blueprint.route("/", methods=["POST"])
def detect():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PREPROCESS_OUTPUT, exist_ok=True)

    aggregated_objects = {}

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        folder = preprocess_image(filepath, PREPROCESS_OUTPUT)
        detections = run_detection_on_folder(folder, model_path=MODEL_PATH)
        objects_count = postprocess_detections(detections)

        for obj_name, count in objects_count.items():
            aggregated_objects[obj_name] = aggregated_objects.get(obj_name, 0) + count

    return jsonify(aggregated_objects)
