# Gesture Music Controller

A real time hand gesture music controller that uses a webcam to detect and classify hand gestures, then streams continuous control signals to a DAW over OSC. Built from scratch in Python - no special hardware required, just a laptop and a webcam.

## What It Does

Hold a gesture, move your hand, and control music in real time:

| Gesture | Axes | Controls |
|---------|------|----------|
| **CLOSED_FIST** | X, Y | Pitch shift + Volume |
| **OPEN_PALM** | X | Filter cutoff |
| **POINT** | X, Y | Reverb depth + Pitch shift |

- Position tracking only activates while a gesture is held, preventing network flooding during idle states.
- Dropping a hand from frame fires a `DROP` event, which can trigger actions like stopping playback.
- Supports two hands simultaneously with a left-hand/right-hand role split for structure vs. expression.

## Architecture

The pipeline is modular - each stage is a separate file with a single responsibility:

```
Webcam -> capture.py -> hand_tracker.py -> geometry.py → smoothing.py -> state_machine.py -> position_tracker.py -> osc_sender.py -> DAW
```

### `capture.py`
Webcam capture via OpenCV. Provides a `CameraStream` class that handles frame acquisition.

### `hand_tracker.py`
Wraps MediaPipe's hand detection. Assigns stable hand IDs by sorting detected hands by wrist X position (leftmost = Hand 0), which proved more reliable than MediaPipe's internal ordering during fast motion. Handles mirror flip for natural interaction.

### `geometry.py`
Computes palm-size-normalised, distance invariant geometry from raw landmarks. Measures four finger flexion using vector dot product angles rather than simple distance thresholds, making gesture recognition work consistently regardless of how far you are from the camera.

### `smoothing.py`
Implements a One Euro Filter for low latency signal smoothing. The `HandSmoother` class isolates filters per hand and per signal using `(hand_index, signal_name)` keys, preventing crosstalk between independent tracking streams.

### `state_machine.py`
A `GestureStateMachine` that classifies finger flexion data into gesture states using empirically derived thresholds:
- Straight finger: >160° entry / <130° exit
- Curled finger: <60°
- Fist cushion: <85°
- Ring/pinky cushion for POINT: <70°

Uses 3 frame debounce and hysteresis dead zones between thresholds to eliminate flickering at gesture boundaries. Fires a `DROP` event when a hand leaves the frame.

### `position_tracker.py`
Converts hand position to normalised axis values using a clutched system - position math only runs while a specific gesture is active. Each gesture maps to its own axes, preventing unintended control changes during gesture transitions.

### `osc_sender.py`
Sends control signals over OSC to localhost port 8000 using `python-osc`. Each gesture+axis combination gets a unique OSC address (e.g. `/hand/0/fist/y` for volume, `/hand/0/palm/y` for filter cutoff), so the receiving DAW can bind each control independently without overlap. Designed to connect to REAPER or any OSC compatible DAW.

## Design Decisions

**Why OSC over MIDI?** OSC matches the real Mi.Mu architecture and supports continuous floating point control values. MIDI's 7 bit resolution (0–127) would quantise the smooth gesture data.

**Why raw geometry instead of a gesture recognition library?** The goal is controllable, real time musical expression - not just gesture classification. Building the geometry and state machine from scratch gives full control over thresholds, debounce timing, and the relationship between hand movement and musical output.

**Why wrist X sorting for hand IDs?** MediaPipe's internal hand ordering is unstable during fast motion, and its Left/Right handedness labels are inverted due to the mirror flip. Sorting by wrist X position provides consistent identification.

**Why One Euro Filter?** It adapts its cutoff frequency based on signal speed - slow movements get heavily smoothed, fast movements pass through with minimal lag. This is critical for musical expression where both precision and responsiveness matter.

**Why hysteresis thresholds?** A single threshold for "finger is straight" would cause rapid flickering when a finger hovers near the boundary. Separate entry (>160°) and exit (<130°) thresholds create a dead zone that eliminates false triggers without adding latency.

**Why gesture-specific OSC addresses?** Different gestures can control the same axis (e.g. CLOSED_FIST Y = volume, OPEN_PALM Y = filter cutoff). Using `/hand/{id}/{gesture}/{axis}` instead of `/hand/{id}/slider/{axis}` gives the receiving DAW a unique address per control, eliminating parameter conflicts when switching gestures.

## Setup

**Requirements:** Python 3.11+, a webcam

```bash
# Clone the repo
git clone https://github.com/tanvi-anand/gesture-music-controller.git
cd gesture-music-controller

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start the controller
python main.py

# In a separate terminal, run the test receiver to verify OSC output
python test_osc_receiver.py
```

For DAW integration, configure your DAW to receive OSC on localhost port 8000. In REAPER, add an OSC control surface under Options > Preferences > Control/OSC/Web, set to "Local port [receive only]" on port 8000, and use FX Learn to bind each gesture address to a plugin parameter.

## Tech Stack

- **Python 3.11** - core language
- **OpenCV** - webcam capture
- **MediaPipe** - hand landmark detection
- **NumPy** - geometry calculations
- **python-osc** - OSC networking

## Known Limitations

- **2D camera only:** Roll, pitch, and yaw axes are infeasible with a single webcam - intentionally excluded from scope.
- **MediaPipe handedness labels are inverted** due to mirror flip - a known accepted limitation. Hand identification relies on wrist-X sorting instead.
- **Single user desktop application:** No multiuser, no cloud, no backend. This is a real time control signal pipeline, not a web service.
- **No original ML research:** The project's strength is the engineering pipeline around MediaPipe, not the hand detection model itself.