import cv2
from src.capture import CameraStream
from src.hand_tracker import HandTracker, HandIdentityTracker
from src.geometry import get_normalized_distance, get_finger_flexion
from src.smoothing import HandSmoother
from src.state_machine import GestureStateMachine, GestureState
from src.position_tracker import PositionTracker
from src.osc_sender import OSCSender

def main():
    camera = CameraStream(camera_index=0)
    tracker = HandTracker()
    identity_tracker = HandIdentityTracker()
    smoother = HandSmoother(min_cutoff=1.0, beta=0.5)

    state_machines = {}
    pos_tracker = PositionTracker()
    osc = OSCSender(ip="127.0.0.1", port=8000)

    print("Webcam started. Press 'q' on your keyboard to quit.")
    print("OSC streaming to 127.0.0.1:8000")

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

                if hand_index not in state_machines:
                    state_machines[hand_index] = GestureStateMachine(debounce_frames=5)

                transition = state_machines[hand_index].update(
                    smooth_index, smooth_middle, smooth_ring, smooth_pinky
                )

                event = None
                if transition is not None:
                    new_gesture = transition.split(" -> ")[1]
                    event = {"type": "ENTER", "gesture": new_gesture}
                    osc.send_gesture(hand_index, new_gesture)
                    print(f">>> Hand {hand_index} GESTURE: {transition}")

                pos = pos_tracker.update(
                    hand_id=hand_index,
                    event=event,
                    landmarks=hand_landmarks.landmark,
                    frame_w=frame.shape[1],
                    frame_h=frame.shape[0],
                )

                # Send each active axis over OSC.
                if pos is not None and pos["active_axes"]:
                    for axis in pos["active_axes"]:
                        value = pos[axis]
                        gesture_name =  GestureState.NAMES[state_machines[hand_index].confirmed_state]
                        osc.send_slider(hand_index, gesture_name, axis, value)

        else:
            for hand_id, machine in state_machines.items():
                drop_event = machine.hand_missing()
                if drop_event is not None:
                    print(f">>> Hand {hand_id} EVENT: {drop_event}")
                    osc.send_event(hand_id, "DROP")

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