import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Load MediaPipe model
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)


def recognize_gesture(hand):

    # Thumb
    thumb_up = hand[4].y < hand[3].y

    # Other fingers
    index_up = hand[8].y < hand[6].y
    middle_up = hand[12].y < hand[10].y
    ring_up = hand[16].y < hand[14].y
    little_up = hand[20].y < hand[18].y

    # Thumbs Up
    if thumb_up and not index_up and not middle_up and not ring_up and not little_up:
        return "THUMBS UP"

    # Open Hand
    if index_up and middle_up and ring_up and little_up:
        return "OPEN HAND"

    # Fist
    if not index_up and not middle_up and not ring_up and not little_up:
        return "FIST"

    # Count fingers
    fingers = sum([
        index_up,
        middle_up,
        ring_up,
        little_up
    ])

    if fingers == 1:
        return "ONE FINGER"

    elif fingers == 2:
        return "TWO FINGERS"

    elif fingers == 3:
        return "THREE FINGERS"

    return "UNKNOWN"


# Open camera
cap = cv2.VideoCapture(0)

print("Starting IntelliMed Gesture Recognition...")


while True:

    success, frame = cap.read()

    if not success:
        print("Cannot read camera")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect hand
    result = landmarker.detect(mp_image)

    gesture = "NO HAND"

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        gesture = recognize_gesture(hand)

        # Draw hand landmarks
        for landmark in hand:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    # Display gesture
    cv2.putText(
        frame,
        "Gesture: " + gesture,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "IntelliMed Gesture Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()