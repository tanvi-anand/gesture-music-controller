"""
Conditional position tracking — "clutched air sliders."
Each gesture gates specific axes. Filters run continuously
so values are warm on engagement.
"""

import time
from src.smoothing import OneEuroFilter


# Which axes each gesture controls.
# If a gesture isn't in this dict, it doesn't clutch any slider.
GESTURE_AXIS_MAP = {
    "CLOSED_FIST": ["y"],          # volume
    "OPEN_PALM":   ["x"],          # reverb depth
    "POINT":       ["x"],          # FX preset
    "PEACE":       ["x", "y"],     # pitch shift + vibrato
}


class PositionTracker:
    """
    Tracks hand position gated by gesture.
    Each gesture activates its designated axes.
    Filters run continuously so values are warm on engagement.
    """

    def __init__(self):
        self._active_gesture = {}
        self._filters = {}
        self._positions = {}

    def update(self, hand_id, event, landmarks, frame_w, frame_h):
        """
        Call once per hand per frame, AFTER the state machine.

        Returns:
            dict with "x", "y" (always computed),
            "active_axes" (list of str, or empty list),
            and "gesture" (str or None).
            Returns None if this hand has never been seen.
        """
        self._update_active_gesture(hand_id, event)
        current_gesture = self._active_gesture.get(hand_id)
        active_axes = GESTURE_AXIS_MAP.get(current_gesture, [])

        if landmarks is not None:
            now = time.perf_counter()
            flt = self._get_filters(hand_id)

            raw_x = landmarks[0].x
            raw_y = 1.0 - landmarks[0].y

            sx = self._clamp01(flt["x"].filter(raw_x, now))
            sy = self._clamp01(flt["y"].filter(raw_y, now))

            self._positions[hand_id] = {
                "x": round(sx, 4),
                "y": round(sy, 4),
                "active_axes": list(active_axes),
                "gesture": current_gesture,
            }
        elif hand_id in self._positions:
            self._positions[hand_id]["active_axes"] = list(active_axes)
            self._positions[hand_id]["gesture"] = current_gesture

        return self._positions.get(hand_id)

    def _update_active_gesture(self, hand_id, event):
        if event is None:
            return

        etype = event.get("type")
        egesture = event.get("gesture")

        if etype == "ENTER":
            if egesture in GESTURE_AXIS_MAP:
                self._active_gesture[hand_id] = egesture
            else:
                self._active_gesture[hand_id] = None
        elif etype in ("EXIT", "DROP"):
            self._active_gesture[hand_id] = None

    def _get_filters(self, hand_id):
        if hand_id not in self._filters:
            self._filters[hand_id] = {
                "x": OneEuroFilter(min_cutoff=0.5, beta=0.007),
                "y": OneEuroFilter(min_cutoff=0.5, beta=0.007),
            }
        return self._filters[hand_id]

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, value))