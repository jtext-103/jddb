from .. import BaseProcessor, Signal
from typing import Tuple


class TrimProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Tuple[Signal, ...]:
        """Trim all signals to the same length with the shortest one.

        Args:
            *signal: The signals to be trimmed.

        Returns: Tuple[Signal, ...]: The signals trimmed.
        """
        lengths = [len(each_signal.data) for each_signal in signal]
        min_length = min(lengths)
        for each_signal in signal:
            each_signal.data = each_signal.data[:min_length]
        return signal
