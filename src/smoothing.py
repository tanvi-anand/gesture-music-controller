import math
import time


class OneEuroFilter:
    # This class handles smoothing for ONE single number over time
    # (e.g., just "Hand 1's index finger flexion angle" and nothing else).
    # We will create many of these, one per signal we want smoothed.

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        # min_cutoff: the baseline amount of smoothing when the value is
        # barely changing. Lower = smoother but more laggy at rest.
        self.min_cutoff = min_cutoff

        # beta: how aggressively the filter loosens up when it detects
        # fast movement. Higher = more responsive during fast motion,
        # but slightly more jitter allowed through during that motion.
        self.beta = beta

        # d_cutoff: a smoothing setting specifically for how we measure
        # "speed" itself, so our speed estimate isn't jittery either.
        self.d_cutoff = d_cutoff

        # These track the filter's memory between frames.
        # They start as None because we haven't received any data yet.
        self.previous_value = None
        self.previous_derivative = None
        self.previous_timestamp = None

    def _smoothing_factor(self, time_elapsed, cutoff):
        # This is part of the standard One Euro Filter formula.
        # It converts a "cutoff" setting and elapsed time into
        # an actual blending weight between 0 and 1.
        r = 2 * math.pi * cutoff * time_elapsed
        return r / (r + 1)

    def _exponential_smoothing(self, smoothing_factor, current_value, previous_value):
        # This blends the current raw value with the previous smoothed value.
        # A smoothing_factor closer to 1 trusts the new value more.
        # A smoothing_factor closer to 0 trusts the old, smoothed value more.
        return (smoothing_factor * current_value) + ((1 - smoothing_factor) * previous_value)

    def filter(self, current_value, current_timestamp=None):
        # If no timestamp is given, just use the current real-world time.
        if current_timestamp is None:
            current_timestamp = time.time()

        # First-ever call: we have nothing to smooth against yet,
        # so we just accept this value as-is and remember it.
        if self.previous_value is None:
            self.previous_value = current_value
            self.previous_derivative = 0.0
            self.previous_timestamp = current_timestamp
            return current_value

        # Figure out how much time passed since the last frame.
        time_elapsed = current_timestamp - self.previous_timestamp

        # Safety check: if time_elapsed is zero or negative (can happen
        # with duplicate timestamps), skip smoothing this frame rather
        # than risk dividing by zero.
        if time_elapsed <= 0:
            return self.previous_value

        # Step 1: estimate how fast the value is currently changing (the "derivative").
        raw_derivative = (current_value - self.previous_value) / time_elapsed

        # Step 2: smooth THAT speed estimate too, so speed itself isn't jittery.
        derivative_smoothing_factor = self._smoothing_factor(time_elapsed, self.d_cutoff)
        smoothed_derivative = self._exponential_smoothing(
            derivative_smoothing_factor,
            raw_derivative,
            self.previous_derivative
        )

        # Step 3: use that smoothed speed to decide how much to trust
        # the new value right now. Faster movement = higher cutoff =
        # less smoothing = more responsiveness.
        adjusted_cutoff = self.min_cutoff + (self.beta * abs(smoothed_derivative))

        # Step 4: actually smooth the real value using that adjusted cutoff.
        value_smoothing_factor = self._smoothing_factor(time_elapsed, adjusted_cutoff)
        smoothed_value = self._exponential_smoothing(
            value_smoothing_factor,
            current_value,
            self.previous_value
        )

        # Remember this frame's results for next time.
        self.previous_value = smoothed_value
        self.previous_derivative = smoothed_derivative
        self.previous_timestamp = current_timestamp

        return smoothed_value


class HandSmoother:
    # This class manages a whole COLLECTION of OneEuroFilters,
    # one for each unique signal, keyed by both which hand it belongs
    # to and which measurement it is (e.g. "hand 0's index flexion").
    # This solves the "don't mix up two hands' data" problem directly.

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # This dictionary will hold our filters. Keys look like
        # (0, "index_flexion") or (1, "pinch_distance") - a combination
        # of hand number and signal name, so each stays independent.
        self.filters = {}

    def smooth(self, hand_index, signal_name, raw_value):
        # Build the unique key for this specific hand + signal combination.
        key = (hand_index, signal_name)

        # If we've never seen this particular combination before,
        # create a brand new filter just for it.
        if key not in self.filters:
            self.filters[key] = OneEuroFilter(
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff
            )

        # Run the raw value through that specific filter and return the result.
        return self.filters[key].filter(raw_value)