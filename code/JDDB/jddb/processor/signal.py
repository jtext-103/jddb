import numpy as np
import warnings


class Signal(object):
    """Signal object.

    Stores data and necessary parameters of a signal.

    Attributes:
        data (numpy.ndarray): raw data of the Signal instance.
        attributes (dict): necessary attributes of the Signal instance. Note that SampleRate and StartTime will be set default if not given.
        tag (str): name of the signal.
    """

    def __init__(self, data: np.ndarray, attributes: dict, tag: str = None):
        self.data = data
        self.attributes = attributes
        self.tag = tag

        if 'SampleRate' not in self.attributes.keys():
            self.attributes['SampleRate'] = 1
            warnings.warn("SampleRate not in attributes. Set to 1.")

        if 'StartTime' not in self.attributes.keys():
            self.attributes['StartTime'] = 0
            warnings.warn("StartTime not in attributes. Set to 0.")

    @property
    def time(self):
        """
        Returns:
            numpy.ndarray: time axis of the Signal instance according to SampleRate and StartTime given in the attributes.
        """
        start_time = self.attributes['StartTime']
        sample_rate = self.attributes['SampleRate']
        down_time = start_time + (len(self.data) - 1) / sample_rate
        return np.linspace(start_time, down_time, len(self.data))
