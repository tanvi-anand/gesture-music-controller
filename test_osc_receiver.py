"""
Test receiver: listens on localhost port 8000 and prints every
OSC message that arrives. Run this in a SEPARATE terminal window
from main.py to prove the full pipeline works end-to-end.

Usage:
    python test_osc_receiver.py

Then in another terminal:
    python main.py

You should see gesture events and slider values printing here
as you make hand gestures in the webcam feed.

Press Ctrl+C to stop.
"""

from pythonosc import dispatcher, osc_server


def handle_message(address, *args):
    """Print every OSC message that arrives, regardless of address."""
    # Format the values nicely — floats get 3 decimal places,
    # strings print as-is.
    formatted = []
    for arg in args:
        if isinstance(arg, float):
            formatted.append(f"{arg:.3f}")
        else:
            formatted.append(str(arg))
    values = "  ".join(formatted)
    print(f"{address}  →  {values}")


def main():
    # Create a dispatcher that routes ALL incoming messages
    # to our single handler function. The "*" means "match
    # every address pattern."
    disp = dispatcher.Dispatcher()
    disp.set_default_handler(handle_message)

    # Start a blocking UDP server on localhost:8000.
    server = osc_server.BlockingOSCUDPServer(("127.0.0.1", 8000), disp)
    print(f"OSC receiver listening on 127.0.0.1:8000")
    print(f"Waiting for messages... (Ctrl+C to stop)\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReceiver stopped.")
        server.server_close()


if __name__ == "__main__":
    main()