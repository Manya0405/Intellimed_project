import cv2
import mediapipe as mp

from app.gesture.gesture_detector import GestureDetector


detector = GestureDetector()

cap = cv2.VideoCapture(0)

print("Gesture module camera test started")


while True:

    success, frame = cap.read()

    if not success:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.landmarker.detect(mp_image)

    gesture = "NO HAND"

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        gesture = detector.recognize_gesture(hand)

    cv2.putText(
        frame,
        "Gesture: " + gesture,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("IntelliMed Gesture Module", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()