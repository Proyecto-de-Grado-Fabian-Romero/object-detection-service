from flasgger import Swagger
from flask import Flask

from app.routes.detect_routes import detect_blueprint
from app.routes.preprocess_routes import preprocess_blueprint
from app.routes.process_routes import process_blueprint


def create_app():
    app = Flask(__name__)

    app.config["SWAGGER"] = {
        "title": "Object Detection API",
        "uiversion": 3,
    }
    Swagger(app)

    app.register_blueprint(preprocess_blueprint, url_prefix="/preprocess")
    app.register_blueprint(detect_blueprint, url_prefix="/detect")
    app.register_blueprint(process_blueprint, url_prefix="/process")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(port=5151, debug=True)
