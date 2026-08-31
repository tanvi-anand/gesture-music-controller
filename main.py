import cv2
from src.capture import CameraStream
from src.hand_tracker import HandTracker, HandIdentityTracker
from src.geometry import get_normalized_distance, get_finger_flexion
from src.smoothing import HandSmoother
from src.state_machine import GestureStateMachine
from src.position_tracker import PositionTracker

def main():
    camera = CameraStream(camera_index=0)
    tracker = HandTracker()
    identity_tracker = HandIdentityTracker()
    smoother = HandSmoother(min_cutoff=1.0, beta=0.5)

    state_machines = {}
    pos_tracker = PositionTracker(z_min_palm_px=45.0, z_max_palm_px=250.0)

    print("Webcam started. Press 'q' on your keyboard to quit.")

    while True:
        frame = camera.get_frame()

        if frame is None:
            print("Warning: failed to grab a frame. Retrying...")
            continue

        frame = cv2.flip(frame, 1)

        frame, results = tracker.find_hands(frame)

        if results.multi_hand_landmarks:
            stable_hands = identity_tracker.assign_stable_ids(results.multi_hand_landmarks)

            for hand_index, hand_landmarks in stable_hands:
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                raw_pinch = get_normalized_distance(thumb_tip, index_tip, hand_landmarks)
                raw_index_flexion = get_finger_flexion(hand_landmarks, "index")
                raw_middle_flexion = get_finger_flexion(hand_landmarks, "middle")
                raw_ring_flexion = get_finger_flexion(hand_landmarks, "ring")
                raw_pinky_flexion = get_finger_flexion(hand_landmarks, "pinky")

                smooth_pinch = smoother.smooth(hand_index, "pinch", raw_pinch)
                smooth_index = smoother.smooth(hand_index, "index_flexion", raw_index_flexion)
                smooth_middle = smoother.smooth(hand_index, "middle_flexion", raw_middle_flexion)
                smooth_ring = smoother.smooth(hand_index, "ring_flexion", raw_ring_flexion)
                smooth_pinky = smoother.smooth(hand_index, "pinky_flexion", raw_pinky_flexion)

                # Create a state machine for this hand if we haven't seen it before.
                if hand_index not in state_machines:
                    state_machines[hand_index] = GestureStateMachine(debounce_frames=5)

                # Feed the smoothed angles into the state machine.
                transition = state_machines[hand_index].update(
                    smooth_index, smooth_middle, smooth_ring, smooth_pinky
                )

                # Bridge: translate the state machine's string output into
                # the event dict format the position tracker expects.
                # Any transition counts as an ENTER into the new gesture.
                # The position tracker internally checks whether that gesture
                # is a clutch gesture or not.
                event = None
                if transition is not None:
                    new_gesture = transition.split(" -> ")[1]
                    event = {"type": "ENTER", "gesture": new_gesture}

                pos = pos_tracker.update(
                    hand_id=hand_index,
                    event=event,
                    landmarks=hand_landmarks.landmark,
                    frame_w=frame.shape[1],
                    frame_h=frame.shape[0],
                )

                # Print gesture transitions.
                if transition is not None:
                    print(f">>> Hand {hand_index} GESTURE: {transition}")

                # Print slider data only when an axis is actively clutched.
                if pos is not None and pos["active_axis"] is not None:
                    axis = pos["active_axis"]
                    value = pos[axis]
                    gesture = pos["gesture"]
                    print(f"  Hand {hand_index} [{gesture}] {axis.upper()}={value:.3f}")

        else:
            # No hands detected this frame — notify all known state machines
            # and the position tracker.
            for hand_id, machine in state_machines.items():
                drop_event = machine.hand_missing()
                if drop_event is not None:
                    print(f">>> Hand {hand_id} EVENT: {drop_event}")

                    pos_tracker.update(
                        hand_id=hand_id,
                        event={"type": "DROP", "gesture": ""},
                        landmarks=None,
                        frame_w=frame.shape[1],
                        frame_h=frame.shape[0],
                    )

        cv2.imshow("Webcam Feed - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()