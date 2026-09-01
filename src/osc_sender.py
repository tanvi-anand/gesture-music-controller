"""
Module E: OSC networking.
Sends gesture events and clutched slider values to a receiving
application over UDP on localhost port 8000.

OSC address schema:
    /hand/{id}/gesture          → string (gesture name on transition)
    /hand/{id}/event            → string ("DROP" on hand exit)
    /hand/{id}/slider/{axis}    → float  (0.0–1.0 clutched axis value)
    /hand/{id}/vibrato          → float  (0.0–1.0 index finger flexion)
"""

from pythonosc import udp_client


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

        Args:
            hand_id:       int (0 or 1)
            gesture_name:  str like "CLOSED_FIST", "OPEN_PALM", etc.
        """
        address = f"/hand/{hand_id}/gesture"
        self.client.send_message(address, gesture_name)

    def send_event(self, hand_id, event_name):
        """
        Send a system event like DROP (hand leaving frame).

        Args:
            hand_id:    int (0 or 1)
            event_name: str like "DROP"
        """
        address = f"/hand/{hand_id}/event"
        self.client.send_message(address, event_name)

    def send_slider(self, hand_id, axis, value):
        """
        Send a continuous slider value for a clutched axis.
        Called every frame while the axis is active.

        Args:
            hand_id: int (0 or 1)
            axis:    str — "x", "y", or "z"
            value:   float — 0.0 to 1.0
        """
        address = f"/hand/{hand_id}/slider/{axis}"
        self.client.send_message(address, float(value))

    def send_vibrato(self, hand_id, value):
        """
        Send the index finger flexion value (vibrato dial).
        Called every frame regardless of gesture state.

        Args:
            hand_id: int (0 or 1)
            value:   float — 0.0 (straight / no vibrato) to 1.0 (fully curled / max vibrato)
        """
        address = f"/hand/{hand_id}/vibrato"
        self.client.send_message(address, float(value))