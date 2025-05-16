from flask import Flask
from flasgger import Swagger
from app.routes.preprocess_routes import preprocess_blueprint  # type: ignore


def create_app():
    app = Flask(__name__)

    app.config["SWAGGER"] = {
        "title": "Object Detection API",
        "uiversion": 3,
    }
    Swagger(app)

    # Register route blueprints
    app.register_blueprint(preprocess_blueprint, url_prefix="/preprocess")
    # app.register_blueprint(detect_blueprint, url_prefix="/detect")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(port=5151, debug=True)
