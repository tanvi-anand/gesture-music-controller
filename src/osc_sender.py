"""
Module E: OSC networking.
Sends gesture events and clutched slider values to a receiving
application over UDP on localhost port 8000.

OSC address schema:
    /hand/{id}/gesture/{gesture} → float 1.0 (fires once on transition)
    /hand/{id}/event/{event}     → float 1.0 (fires once, e.g. DROP)
    /hand/{id}/{gesture}/{axis}  → float 0.0–1.0 (continuous slider)

Every message type gets its own unique address so the receiving
application (e.g. REAPER) can bind each one independently.
Gesture transitions and events send 1.0 as a trigger pulse —
REAPER's action system only needs to see "a message arrived"
to fire the bound action.
"""

from pythonosc import udp_client


# Map from state machine gesture names to short OSC-friendly names.
# Keeps addresses compact: /hand/0/fist/y instead of /hand/0/CLOSED_FIST/y
GESTURE_OSC_NAMES = {
    "CLOSED_FIST": "fist",
    "OPEN_PALM":   "palm",
    "POINT":       "point",
}


class OSCSender:
    """
    Wraps python-osc's UDP client and provides named methods
    for each type of message this project sends.
    """

    def __init__(self, ip="127.0.0.1", port=8000):
        """
        Args:
            ip:   target IP address (localhost by default).
            port: target UDP port (8000 by default, matching our spec).
        """
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.ip = ip
        self.port = port

    def send_gesture(self, hand_id, gesture_name):
        """
        Send a discrete gesture transition event.
        Called once per transition, not every frame.

        Each gesture gets its own address so the receiving application
        can bind different actions to different gestures.  Sends 1.0
        as a trigger pulse — REAPER's action binding only needs to
        see "a message arrived at this address" to fire.

        Args:
            hand_id:       int (0 or 1)
            gesture_name:  str like "CLOSED_FIST", "OPEN_PALM", etc.
        """
        short_name = GESTURE_OSC_NAMES.get(gesture_name, gesture_name.lower())
        address = f"/hand/{hand_id}/gesture/{short_name}"
        self.client.send_message(address, 1.0)

    def send_event(self, hand_id, event_name):
        """
        Send a system event like DROP (hand leaving frame).

        Args:
            hand_id:    int (0 or 1)
            event_name: str like "DROP"
        """
        address = f"/hand/{hand_id}/event/{event_name.lower()}"
        self.client.send_message(address, 1.0)

    def send_slider(self, hand_id, gesture_name, axis, value):
        """
        Send a continuous slider value for a clutched axis.
        Called every frame while the axis is active.

        The address includes the gesture name so that the same axis
        controlled by different gestures produces different OSC addresses.
        For example, CLOSED_FIST + Y = /hand/0/fist/y (volume)
        and OPEN_PALM + Y = /hand/0/palm/y (reverb depth).

        Args:
            hand_id:       int (0 or 1)
            gesture_name:  str like "CLOSED_FIST" — converted to short
                           form ("fist") via GESTURE_OSC_NAMES.
            axis:          str "x" or "y"
            value:         float 0.0 to 1.0
        """
        short_name = GESTURE_OSC_NAMES.get(gesture_name, gesture_name.lower())
        address = f"/hand/{hand_id}/{short_name}/{axis}"
        self.client.send_message(address, float(value))