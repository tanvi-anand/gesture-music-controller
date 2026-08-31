import cv2
from src.capture import CameraStream
from src.hand_tracker import HandTracker, HandIdentityTracker
from src.geometry import get_normalized_distance, get_finger_flexion
from src.smoothing import HandSmoother
from src.state_machine import GestureStateMachine

def main():
    camera = CameraStream(camera_index=0)
    tracker = HandTracker()
    identity_tracker = HandIdentityTracker()
    smoother = HandSmoother(min_cutoff=1.0, beta=0.5)

    state_machines = {}

    print("Webcam started. Press 'q' on your keyboard to quit.")

    while True:
        frame = camera.get_frame()

        if frame is None:
            print("Warning: failed to grab a frame. Retrying...")
            continue

        frame = cv2.flip(frame, 1)

        frame, results = tracker.find_hands(frame)

        if results.multi_hand_landmarks:
            # enumerate() gives us both the hand's position in the list (0 or 1)
            # AND the hand's landmark data, in one go. We need that index
            # number specifically to keep each hand's filters separate.
            stable_hands = identity_tracker.assign_stable_ids(results.multi_hand_landmarks)

            for hand_index, hand_landmarks in stable_hands:
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                raw_pinch = get_normalized_distance(thumb_tip, index_tip, hand_landmarks)
                raw_index_flexion = get_finger_flexion(hand_landmarks, "index")
                raw_middle_flexion = get_finger_flexion(hand_landmarks, "middle")
                raw_ring_flexion = get_finger_flexion(hand_landmarks, "ring")
                raw_pinky_flexion = get_finger_flexion(hand_landmarks, "pinky")

                # Run every raw value through the smoother before we use it.
                # Each call passes which hand this is, so hand 0 and hand 1
                # never share a filter's memory.
                smooth_pinch = smoother.smooth(hand_index, "pinch", raw_pinch)
                smooth_index = smoother.smooth(hand_index, "index_flexion", raw_index_flexion)
                smooth_middle = smoother.smooth(hand_index, "middle_flexion", raw_middle_flexion)
                smooth_ring = smoother.smooth(hand_index, "ring_flexion", raw_ring_flexion)
                smooth_pinky = smoother.smooth(hand_index, "pinky_flexion", raw_pinky_flexion)

                # Create a state machine for this hand if we haven't seen it before.
                if hand_index not in state_machines:
                    state_machines[hand_index] = GestureStateMachine(debounce_frames=5)

                # Feed the smoothed angles into the state machine.
                print(f"Hand {hand_index} ANGLES | Index: {smooth_index:.1f}  Middle: {smooth_middle:.1f}  Ring: {smooth_ring:.1f}  Pinky: {smooth_pinky:.1f}")
                transition = state_machines[hand_index].update(
                    smooth_index, smooth_middle, smooth_ring, smooth_pinky
                )

                # Only print when something actually changes — not every frame.
                if transition is not None:
                    print(f">>> Hand {hand_index} GESTURE: {transition}")

        else:
            # No hands detected this frame — notify all known state machines.
            for hand_id, machine in state_machines.items():
                drop_event = machine.hand_missing()
                if drop_event is not None:
                    print(f">>> Hand {hand_id} EVENT: {drop_event}")

        cv2.imshow("Webcam Feed - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()