from .. import BaseProcessor, Signal
from scipy.interpolate import interp1d
from copy import deepcopy
import numpy as np


class ResamplingProcessor(BaseProcessor):
    def __init__(self, sample_rate: float):
        super().__init__()
        self._sample_rate = sample_rate

    def transform(self, signal: Signal) -> Signal:
        """Resample signals according to the new sample rate given by the processor initialization.

        Args:
            signal: The signal to be resampled.

        Returns: Signal: The resampled signal.

        """
        start_time = signal.time[0]
        end_time = signal.time[-1]
        new_end_time = start_time + int((end_time - start_time) * self._sample_rate) / self._sample_rate
        # new time axis should include start time point
        new_time = np.linspace(start_time, new_end_time,
                               int((new_end_time - start_time) * self._sample_rate) + 1)
        f = interp1d(signal.time, signal.data)
        resampled_attributes = deepcopy(signal.attributes)
        resampled_attributes['SampleRate'] = self._sample_rate

        return Signal(data=f(new_time), attributes=resampled_attributes)
