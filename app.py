from flask import Flask, render_template, request
import os
import cv2
import pytesseract
from difflib import SequenceMatcher
from ultralytics import YOLO


app = Flask(__name__)


# ==========================================================
# TESSERACT OCR
# ==========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==========================================================
# FOLDERS
# ==========================================================

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/uploads/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ==========================================================
# LOAD MODELS
# ==========================================================

# Pill detection model
MODEL_PATH = "runs/detect/train-3/weights/best.pt"

# Medicine classification model
CLASSIFIER_PATH = "runs/classify/train-2/weights/best.pt"

model = YOLO(MODEL_PATH)
classifier = YOLO(CLASSIFIER_PATH)


# ==========================================================
# OCR FUNCTION
# ==========================================================

def read_medicine_text(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Increase size
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2
    )

    # Improve contrast
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # OCR
    text = pytesseract.image_to_string(
        thresh,
        config="--psm 6"
    )

    return text.strip()


# ==========================================================
# OCR MEDICINE MATCHING
# ==========================================================

def find_medicine_from_ocr(ocr_text):

    if not ocr_text:
        return None, 0

    # All medicine names known by our classifier
    medicine_names = list(
        classifier.names.values()
    )

    # Convert OCR text to lowercase
    text = ocr_text.lower()

    # Remove unwanted characters
    cleaned = ""

    for ch in text:

        if ch.isalnum() or ch in " -":

            cleaned += ch

        else:

            cleaned += " "

    text = cleaned

    # ======================================================
    # SPECIFIC OCR CORRECTIONS
    # ======================================================

    # Dolo-650 can be read by OCR as:
    #
    # Bole-650
    # Bole 650
    # Dole-650
    # Dolo 650
    #
    # Check the complete OCR text.

    if "650" in text:

        dolo_words = [
            "dolo",
            "bole",
            "dole",
            "b0le",
            "d0lo"
        ]

        for word in dolo_words:

            if word in text:

                return "Dolo-650", 95.0


    # ======================================================
    # GENERAL FUZZY MATCHING
    # ======================================================

    best_name = None
    best_score = 0

    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Compare complete OCR line
        for medicine in medicine_names:

            medicine_clean = medicine.lower()

            score = SequenceMatcher(
                None,
                line,
                medicine_clean
            ).ratio()

            if score > best_score:

                best_score = score
                best_name = medicine


        # Compare individual words
        words = line.split()

        for word in words:

            for medicine in medicine_names:

                medicine_clean = medicine.lower()

                score = SequenceMatcher(
                    None,
                    word,
                    medicine_clean
                ).ratio()

                if score > best_score:

                    best_score = score
                    best_name = medicine


    # Accept reasonably strong match
    if best_score >= 0.70:

        return (
            best_name,
            round(best_score * 100, 2)
        )

    return None, 0


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# GESTURE
# ==========================================================

@app.route("/gesture")
def gesture():

    return render_template(
        "gesture.html"
    )


# ==========================================================
# YOLO TEST
# ==========================================================

@app.route("/yolo-test")
def yolo_test():

    return "YOLO test page is working."


# ==========================================================
# OCR TEST
# ==========================================================

@app.route("/ocr-test")
def ocr_test():

    return "OCR test page is working."


# ==========================================================
# MEDICINE
# ==========================================================

@app.route(
    "/medicine",
    methods=["GET", "POST"]
)
def medicine():

    # ------------------------------------------------------
    # OPEN PAGE
    # ------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "medicine/index.html"
        )


    # ------------------------------------------------------
    # CHECK FILE
    # ------------------------------------------------------

    if "image" not in request.files:

        return "No image uploaded."


    file = request.files["image"]


    if file.filename == "":

        return "No image selected."


    # ------------------------------------------------------
    # SAVE IMAGE
    # ------------------------------------------------------

    filename = file.filename

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(image_path)


    # ======================================================
    # 1. OCR
    # ======================================================

    ocr_text = read_medicine_text(
        image_path
    )


    # Try to identify medicine from OCR
    ocr_medicine, ocr_confidence = (
        find_medicine_from_ocr(
            ocr_text
        )
    )


    # ======================================================
    # 2. PILL DETECTION
    # ======================================================

    results = model(
        image_path,
        conf=0.30,
        iou=0.50
    )

    result = results[0]

    count = len(
        result.boxes
    )


    # ------------------------------------------------------
    # SAVE DETECTION IMAGE
    # ------------------------------------------------------

    result_filename = (
        "result_" + filename
    )

    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )

    result.save(
        filename=result_path
    )


    # ======================================================
    # 3. MEDICINE CLASSIFIER
    # ======================================================

    classification = classifier(
        image_path
    )

    probs = classification[0].probs


    classifier_name = (
        classification[0].names[
            probs.top1
        ]
    )


    classifier_confidence = float(
        probs.top1conf
    )


    classifier_confidence_percent = round(
        classifier_confidence * 100,
        2
    )


    # ======================================================
    # 4. CHOOSE FINAL RESULT
    # ======================================================

    if ocr_medicine is not None:

        # OCR found a medicine name
        medicine_name = ocr_medicine

        confidence_percent = (
            ocr_confidence
        )

        detection_source = "OCR"


    else:

        # OCR could not identify medicine,
        # so use classifier result.

        medicine_name = classifier_name

        confidence_percent = (
            classifier_confidence_percent
        )

        detection_source = "AI Classifier"


    # ======================================================
    # 5. SEND RESULT TO HTML
    # ======================================================

    return render_template(

        "medicine/index.html",

        image=result_filename,

        count=count,

        medicine_name=medicine_name,

        confidence=confidence_percent,

        ocr_text=ocr_text,

        detection_source=detection_source,

        classifier_name=classifier_name,

        classifier_confidence=(
            classifier_confidence_percent
        )
    )


# ==========================================================
# MEDICINE INFO
# ==========================================================

@app.route(
    "/medicine-info",
    methods=["GET", "POST"]
)
def medicine_info():

    return "Medicine information page is working."


# ==========================================================
# GESTURE CAMERA
# ==========================================================

@app.route("/gesture-camera")
def gesture_camera():

    return "Gesture camera page is working."


# ==========================================================
# YOLO CAMERA
# ==========================================================

@app.route("/yolo-camera")
def yolo_camera():

    return "YOLO camera page is working."


# ==========================================================
# COMBINED CAMERA
# ==========================================================

@app.route("/combined-camera")
def combined_camera():

    return "Combined camera page is working."


# ==========================================================
# OCR CAMERA
# ==========================================================

@app.route("/ocr-camera")
def ocr_camera():

    return "OCR camera page is working."


# ==========================================================
# START APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )