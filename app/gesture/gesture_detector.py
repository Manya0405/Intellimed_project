from pathlib import Path
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GestureDetector:

    def __init__(self):

        model_path = Path(__file__).resolve().parents[2] / "hand_landmarker.task"

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )

        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )

    def recognize_gesture(self, hand):

        thumb_up = hand[4].y < hand[3].y

        index_up = hand[8].y < hand[6].y
        middle_up = hand[12].y < hand[10].y
        ring_up = hand[16].y < hand[14].y
        little_up = hand[20].y < hand[18].y

        if thumb_up and not index_up and not middle_up and not ring_up and not little_up:
            return "THUMBS UP"

        if index_up and middle_up and ring_up and little_up:
            return "OPEN HAND"

        if not index_up and not middle_up and not ring_up and not little_up:
            return "FIST"

        fingers = sum([
            index_up,
            middle_up,
            ring_up,
            little_up
        ])

        if fingers == 1:
            return "ONE FINGER"

        if fingers == 2:
            return "TWO FINGERS"

        if fingers == 3:
            return "THREE FINGERS"

        return "UNKNOWN"