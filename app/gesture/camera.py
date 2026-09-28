import cv2
from app.gesture.gesture_detector import GestureDetector


detector = GestureDetector()


def generate_frames():

    camera = cv2.VideoCapture(0)

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Convert image for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand
        import mediapipe as mp

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = detector.landmarker.detect(mp_image)

        gesture_name = "NO HAND"

        if result.hand_landmarks:
            gesture_name = detector.recognize_gesture(
                result.hand_landmarks[0]
            )

        # Display gesture on camera
        cv2.putText(
            frame,
            gesture_name,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Encode frame
        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

    camera.release()