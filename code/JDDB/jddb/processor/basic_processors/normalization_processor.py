from .. import BaseProcessor, Signal
import numpy as np


class NormalizationProcessor(BaseProcessor):
    def __init__(self, std: float, mean: float):
        super().__init__()
        self._std = std
        self._mean = mean

    def transform(self, signal: Signal) -> Signal:
        """Compute the z-score.
        Compute the z-score of the given signal according to the param_dict given by the processor initialization.

        Note:
            The result of the normalized to signal will be clipped to [-10, 10] if beyond the range.

        Args:
            signal: The signal to be normalized.

        Returns: Signal: The normalized signal.
        """
        normalized_data = (signal.data - self._mean) / self._std
        normalized_data = np.clip(normalized_data, -10, 10)

        return Signal(data=normalized_data, attributes=signal.attributes)
