import numpy as np
from typing import Optional
import warnings


class Signal(object):
    """Signal object.

    Stores data and necessary parameters of a signal.

    Attributes:
        data (numpy.ndarray): raw data of the Signal instance.
        prop_dict (dict): necessary attributes of the Signal instance. Note that SampleRate and StartTime will be set default if not given.
        tag (str): name of the signal. Optional.
    """

    def __init__(self, data: np.ndarray, prop_dict, tag: Optional[str] = None):
        self.data = data
        self.prop_dict = prop_dict
        self.tag = tag

        if 'SampleRate' not in self.prop_dict.keys():
            self.prop_dict['SampleRate'] = 1
            warnings.warn("SampleRate not in props. Set to 1.")

        if 'StartTime' not in self.prop_dict.keys():
            self.prop_dict['StartTime'] = 0
            warnings.warn("StartTime not in props. Set to 0.")

    @property
    def time(self):
        """
        Returns:
            numpy.ndarray: time axis of the Signal instance according to SampleRate and StartTime given in the prop_dict.
        """
        start_time = self.prop_dict['StartTime']
        sample_rate = self.prop_dict['SampleRate']
        down_time = start_time + (len(self.data) - 1) / sample_rate
        return np.linspace(start_time, down_time, len(self.data))
