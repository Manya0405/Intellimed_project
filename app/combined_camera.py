import cv2
from ultralytics import YOLO
import mediapipe as mp

from app.gesture.gesture_detector import GestureDetector


model = YOLO("yolo11n.pt")
gesture_detector = GestureDetector()


def generate_combined_frames():

    camera = cv2.VideoCapture(0)

    while True:

        success, frame = camera.read()

        if not success:
            break

        # =========================
        # YOLO DETECTION
        # =========================

        results = model(frame, verbose=False)

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                name = model.names[class_id]

                label = f"{name} ({confidence:.2f})"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # =========================
        # GESTURE DETECTION
        # =========================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = gesture_detector.landmarker.detect(mp_image)

        gesture_name = "NO HAND"

        if result.hand_landmarks:

            gesture_name = gesture_detector.recognize_gesture(
                result.hand_landmarks[0]
            )

        # Display gesture
        cv2.putText(
            frame,
            "Gesture: " + gesture_name,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        # Encode frame
        ret, buffer = cv2.imencode(".jpg", frame)

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

    camera.release()