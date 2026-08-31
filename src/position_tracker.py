"""
Conditional position tracking — "clutched air sliders."
Each gesture gates a specific axis. Filters run continuously
so values are warm on engagement.
"""

import time
from src.smoothing import OneEuroFilter


# Which axis each gesture controls.
# If a gesture isn't in this dict, it doesn't clutch any slider.
GESTURE_AXIS_MAP = {
    "CLOSED_FIST": "y",     # arm up/down = volume
    "OPEN_PALM":   "z",     # near/far = reverb depth
    "POINT":       "x",     # left/right = FX preset
    "PEACE":       "x",     # left/right = pitch shift
}


class PositionTracker:
    """
    Tracks hand position gated by gesture.
    Each gesture only activates its designated axis.
    Filters run continuously so values are warm on engagement.
    """

    def __init__(self, z_min_palm_px=50.0, z_max_palm_px=250.0):
        """
        Args:
            z_min_palm_px:   expected palm pixel size when hand is far from camera.
            z_max_palm_px:   expected palm pixel size when hand is close to camera.
        """
        self.z_min = z_min_palm_px
        self.z_max = z_max_palm_px

        # Per-hand internal state
        self._active_gesture = {}   # hand_id -> gesture name or None
        self._filters = {}          # hand_id -> dict of OneEuroFilters
        self._positions = {}        # hand_id -> latest position dict

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def update(self, hand_id, event, landmarks, frame_w, frame_h):
        """
        Call once per hand per frame, AFTER the state machine.

        Args:
            hand_id:    int (0 or 1)
            event:      dict {"type": "ENTER", "gesture": str}
                        or {"type": "DROP", "gesture": ""}
                        or None if no transition happened this frame
            landmarks:  MediaPipe landmark list for this hand, or None
            frame_w:    frame width in pixels
            frame_h:    frame height in pixels

        Returns:
            dict with "x", "y", "z" (always computed),
            "active_axis" (str or None), and "gesture" (str or None).
            Returns None if this hand has never been seen.
        """
        # --- 1. Update which gesture (if any) is clutching ---
        self._update_active_gesture(hand_id, event)
        current_gesture = self._active_gesture.get(hand_id)
        active_axis = GESTURE_AXIS_MAP.get(current_gesture)

        # --- 2. Always compute + smooth position (keeps filters warm) ---
        if landmarks is not None:
            now = time.perf_counter()
            flt = self._get_filters(hand_id)

            raw_x = landmarks[0].x                       # 0.0 left -> 1.0 right
            raw_y = 1.0 - landmarks[0].y                 # flip so 0.0 bottom -> 1.0 top
            raw_z = self._palm_to_depth(landmarks, frame_w, frame_h)

            sx = self._clamp01(flt["x"].filter(raw_x, now))
            sy = self._clamp01(flt["y"].filter(raw_y, now))
            sz = self._clamp01(flt["z"].filter(raw_z, now))

            self._positions[hand_id] = {
                "x": round(sx, 4),
                "y": round(sy, 4),
                "z": round(sz, 4),
                "active_axis": active_axis,
                "gesture": current_gesture,
            }
        elif hand_id in self._positions:
            # Hand disappeared — keep last position, update flags
            self._positions[hand_id]["active_axis"] = active_axis
            self._positions[hand_id]["gesture"] = current_gesture

        return self._positions.get(hand_id)

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _update_active_gesture(self, hand_id, event):
        """Track which gesture is currently held, if any."""
        if event is None:
            return

        etype = event.get("type")
        egesture = event.get("gesture")

        if etype == "ENTER":
            # If the new gesture is in our axis map, it's a clutch gesture.
            # If it's NOT (e.g. UNKNOWN), that means we left the clutch.
            if egesture in GESTURE_AXIS_MAP:
                self._active_gesture[hand_id] = egesture
            else:
                self._active_gesture[hand_id] = None
        elif etype in ("EXIT", "DROP"):
            self._active_gesture[hand_id] = None

    def _get_filters(self, hand_id):
        """Lazily create a set of three OneEuroFilters for a hand."""
        if hand_id not in self._filters:
            self._filters[hand_id] = {
                "x": OneEuroFilter(min_cutoff=0.5, beta=0.007),
                "y": OneEuroFilter(min_cutoff=0.5, beta=0.007),
                "z": OneEuroFilter(min_cutoff=0.3, beta=0.005),
            }
        return self._filters[hand_id]

    def _palm_to_depth(self, landmarks, frame_w, frame_h):
        """Convert palm pixel size into a 0-1 depth estimate."""
        wrist = landmarks[0]
        mid_base = landmarks[9]

        dx = (wrist.x - mid_base.x) * frame_w
        dy = (wrist.y - mid_base.y) * frame_h
        palm_px = (dx ** 2 + dy ** 2) ** 0.5

        z_range = self.z_max - self.z_min
        if z_range <= 0:
            return 0.5
        return (palm_px - self.z_min) / z_range

    @staticmethod
    def _clamp01(value):
        """Clamp a value to the 0.0-1.0 range."""
        return max(0.0, min(1.0, value))