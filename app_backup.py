from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)

model = YOLO("runs/detect/train/weights/best.pt")
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/medicine", methods=["GET", "POST"])
def medicine():

    result = None
    image_path = None

    if request.method == "POST":

        if "image" not in request.files:
            result = "No image selected"
            return render_template(
                "medicine/index.html",
                result=result
            )

        image = request.files["image"]

        if image.filename == "":
            result = "No image selected"
            return render_template(
                "medicine/index.html",
                result=result
            )

        image_path = os.path.join(
            UPLOAD_FOLDER,
            image.filename
        )

        image.save(image_path)

        results = model(image_path)

        detected = []

        for r in results:

            for box in r.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                detected.append(
                    f"{class_name} ({confidence:.2f})"
                )

        if detected:
            result = ", ".join(detected)
        else:
            result = "No object detected"

    return render_template(
        "medicine/index.html",
        result=result,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)