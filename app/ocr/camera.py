import cv2

from app.ocr.ocr_reader import read_text
from app.ocr.medicine import process_medicine_text
from app.ocr.medicine_info import get_medicine_info


def generate_ocr_frames():

    camera = cv2.VideoCapture(0)

    while True:

        success, frame = camera.read()

        if not success:
            break

        # OCR
        text = read_text(frame)

        # Medicine processing
        result = process_medicine_text(text)

        medicine_name = result["medicine"]

        # Medicine information
        info = get_medicine_info(medicine_name)

        # Show raw OCR result
        cv2.putText(
            frame,
            "OCR: " + text[:50],
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        if medicine_name != "No medicine detected":

            cv2.putText(
                frame,
                "Medicine: " + info["name"],
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Use: " + info["use"][:55],
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Form: " + info["form"],
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "Medicine not recognized",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

    camera.release()