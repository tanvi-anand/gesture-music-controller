import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.mp_drawing = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

        return frame, results

class HandIdentityTracker:
    # Instead of complex frame-to-frame position matching, we use a
    # dead-simple rule: the hand on the left side of the frame is
    # always hand 0, the hand on the right side is always hand 1.
    # Since we mirrored the camera, left-on-screen = your actual left hand.

    def assign_stable_ids(self, multi_hand_landmarks):
        if not multi_hand_landmarks:
            return []

        # Build a list of (wrist_x_position, hand_landmarks) so we can
        # sort by horizontal position.
        hands_with_positions = []
        for hand_landmarks in multi_hand_landmarks:
            wrist_x = hand_landmarks.landmark[0].x
            hands_with_positions.append((wrist_x, hand_landmarks))

        # Sort by x position: smallest x (leftmost on screen) first.
        hands_with_positions.sort(key=lambda pair: pair[0])

        # Assign IDs based on sorted order:
        # leftmost hand = 0, rightmost hand = 1.
        stable_hands = []
        for position_index, (wrist_x, hand_landmarks) in enumerate(hands_with_positions):
            stable_hands.append((position_index, hand_landmarks))

        return stable_hands