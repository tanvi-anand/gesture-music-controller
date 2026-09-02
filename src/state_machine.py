class GestureState:
    # These are just named constants so we can write readable code
    # like "if state == GestureState.FIST" instead of "if state == 2".
    UNKNOWN = 0
    OPEN_PALM = 1
    CLOSED_FIST = 2
    POINT = 3

    NAMES = {
        0: "UNKNOWN",
        1: "OPEN_PALM",
        2: "CLOSED_FIST",
        3: "POINT",
    }


class GestureStateMachine:
    # One of these exists per hand. It takes in flexion angles every frame,
    # applies threshold + debounce logic, and reports back only when
    # a genuine state transition happens.

    def __init__(self, debounce_frames=3):
        self.confirmed_state = GestureState.UNKNOWN
        self.candidate_state = GestureState.UNKNOWN
        self.candidate_count = 0
        self.debounce_frames = debounce_frames
        self.frames_missing = 0
        self.drop_threshold = 5
        self.drop_fired = False

        # Track which fingers we currently consider "straight" or "curled",
        # so hysteresis can use different thresholds depending on current belief.
        self.finger_straight = {"index": False, "middle": False, "ring": False, "pinky": False}
        self.finger_curled = {"index": False, "middle": False, "ring": False, "pinky": False}

    def _update_finger_states(self, index_angle, middle_angle, ring_angle, pinky_angle):
        # Apply hysteresis: once a finger is considered "straight," it has to
        # drop noticeably below the threshold before we change our mind, and
        # vice versa for "curled." This prevents rapid flickering when an angle
        # hovers right near a boundary.

        angles = {
            "index": index_angle,
            "middle": middle_angle,
            "ring": ring_angle,
            "pinky": pinky_angle,
        }

        for finger, angle in angles.items():
            # Straight detection with hysteresis
            if self.finger_straight[finger]:
                # Currently believed to be straight — only revoke if it drops well below
                if angle < 130:
                    self.finger_straight[finger] = False
            else:
                # Currently not straight — only grant if it clearly passes the threshold
                if angle > 170:
                    self.finger_straight[finger] = True

            # Curled detection with hysteresis
            if self.finger_curled[finger]:
                # Currently believed to be curled — only revoke if it rises well above
                if angle > 90:
                    self.finger_curled[finger] = False
            else:
                # Currently not curled — only grant if it clearly passes the threshold
                if angle < 65:
                    self.finger_curled[finger] = True

    def _classify_raw(self, index_angle, middle_angle, ring_angle, pinky_angle):
        # First, update our hysteresis-tracked finger states.
        self._update_finger_states(index_angle, middle_angle, ring_angle, pinky_angle)

        index_straight = self.finger_straight["index"]
        middle_straight = self.finger_straight["middle"]
        ring_straight = self.finger_straight["ring"]
        pinky_straight = self.finger_straight["pinky"]

        index_curled = self.finger_curled["index"]
        middle_curled = self.finger_curled["middle"]
        ring_curled = self.finger_curled["ring"]
        pinky_curled = self.finger_curled["pinky"]

        # OPEN_PALM: all four fingers straight.
        if index_straight and middle_straight and ring_straight and pinky_straight:
            return GestureState.OPEN_PALM

        # POINT: index straight, other three curled.
        if index_straight and middle_curled and ring_curled and pinky_curled:
            return GestureState.POINT

        # CLOSED_FIST: middle, ring, pinky all curled.
        # Index gets extra cushion because thumb occlusion during a fist
        # makes MediaPipe's index tracking unreliable.
        index_fist_ok = index_angle < 85
        pinky_fist_ok = pinky_angle < 85
        if index_fist_ok and middle_curled and ring_curled and pinky_curled:
            return GestureState.CLOSED_FIST

        return GestureState.UNKNOWN

    def update(self, index_angle, middle_angle, ring_angle, pinky_angle):
        # Called every frame with the latest (smoothed) flexion angles.
        # Returns a transition event string if a state change just happened,
        # or None if nothing changed this frame.

        # Hand is visible this frame, so reset the missing counter.
        self.frames_missing = 0
        self.drop_fired = False

        # Step 1: get the raw classification for this frame.
        raw_state = self._classify_raw(index_angle, middle_angle, ring_angle, pinky_angle)

        # Step 2: debounce logic.
        if raw_state == self.candidate_state:
            # Same state as last frame's candidate — keep counting.
            self.candidate_count += 1
        else:
            # Different state — reset the counter and start watching
            # this new candidate instead.
            self.candidate_state = raw_state
            self.candidate_count = 1

        # Step 3: has the candidate been held long enough to confirm?
        if self.candidate_count >= self.debounce_frames:
            if self.candidate_state != self.confirmed_state:
                # This is a genuine transition: the candidate has been
                # stable for enough frames AND it's different from what
                # we previously confirmed.
                old_state = self.confirmed_state
                self.confirmed_state = self.candidate_state

                old_name = GestureState.NAMES[old_state]
                new_name = GestureState.NAMES[self.confirmed_state]
                return f"{old_name} -> {new_name}"

        # No transition happened this frame.
        return None

    def hand_missing(self):
        # Called when this hand is NOT detected in the current frame.
        # Returns a "drop" event string if the hand has been gone long enough,
        # or None otherwise.

        self.frames_missing += 1

        if self.frames_missing >= self.drop_threshold and not self.drop_fired:
            self.drop_fired = True
            self.confirmed_state = GestureState.UNKNOWN
            self.candidate_state = GestureState.UNKNOWN
            self.candidate_count = 0
            return "DROP"

        return None