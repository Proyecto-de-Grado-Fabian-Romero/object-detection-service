from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
from app.usecases.preprocess_equirect import preprocess_image

preprocess_blueprint = Blueprint("preprocess", __name__)

UPLOAD_FOLDER = "temp_uploads"


@preprocess_blueprint.route("/", methods=["POST"])
@preprocess_blueprint.route("/", methods=["POST"])
def preprocess():
    """
    Preprocess 360º equirectangular images

    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: files
        type: file
        required: true
        description: One or more 360º .jpg or .png images
    responses:
      200:
        description: Preprocessing result
        schema:
          type: array
          items:
            type: object
            properties:
              input_file:
                type: string
              output_path:
                type: string
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    results = []
    for file in files:
        filename = secure_filename(file.filename)
        upload_dir = "temp_uploads"
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        output_dir = preprocess_image(filepath, "output_views")
        results.append({"input_file": filename, "output_path": output_dir})

    return jsonify(results)
