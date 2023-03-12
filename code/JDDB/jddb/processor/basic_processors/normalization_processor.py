from .. import BaseProcessor, Signal
import numpy as np


class NormalizationProcessor(BaseProcessor):
    def __init__(self, param_dict):
        super().__init__()
        self.params.update({"ParamDict": param_dict})

    def transform(self, signal: Signal) -> Signal:
        """Compute the z-score.
        Compute the z-score of the given signal according to the param_dict given by the processor initialization.

        Note:
            The result of the normalized to signal will be clipped to [-10, 10] if beyond the range.

        Args:
            signal: The signal to be normalized.

        Returns: Signal: The normalized signal.
        """
        mean = self.params['ParamDict'][signal.tag][0]
        std = self.params['ParamDict'][signal.tag][1]
        normalized_data = (signal.data - mean) / std
        normalized_data = np.clip(normalized_data, -10, 10)

        return Signal(data=normalized_data, attributes=signal.attributes)
