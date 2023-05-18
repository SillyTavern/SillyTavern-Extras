from flask import make_response, Blueprint, Response, Flask, request
from PIL import Image
from io import BytesIO
from support import Capture


capture_app = Blueprint("capture_app", __name__)


def image_response(image: Image) -> Response:
    byte_stream = BytesIO()
    image.save(byte_stream, format="PNG")
    byte_stream.seek(0)

    response = make_response(byte_stream.getvalue())
    response.headers["Content-Type"] = "image/png"

    return response


@capture_app.route("/generate", methods=["POST"])
def generate():
    # image = Image.new("RGB", (100, 100), color="red")
    # response = image_response(image)
    # return response
    if not request.files:
        return Response("No files found", status=400)
    if "logs" not in request.files:
        return Response("No log file found", status=400)
    user = request.files["user"] if "user" in request.files else None
    char = request.files["char"] if "char" in request.files else None
    capture = Capture(request.files["logs"].read().decode("utf-8"), user, char)

    response = image_response(capture.prepare_image())
    return response


if __name__ == "__main__":
    app = Flask(__name__)
    app.register_blueprint(capture_app, url_prefix="/api/capture")
    app.run(debug=True)
