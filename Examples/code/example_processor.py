import math
from jddb.file_repo import FileRepo
from jddb.processor import Signal, Shot, ShotSet, BaseProcessor
from typing import Optional, List
from scipy.fftpack import fft
from copy import deepcopy
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from scipy import signal as sig
from copy import deepcopy


# class TimeClipProcessor(BaseProcessor):
#
#     """
#         Given the start and end times, output the clipped timing signal
#     """
#     def __init__(self, start_time, end_time=None):
#         super().__init__()
#         self.params.update({
#             "StartTime": start_time,
#             "EndTime": end_time
#         })
#
#     def transform(self, signal: Signal) -> Signal:
#         start_time = self.params['StartTime']
#         if self.params['EndTime']:
#             end_time = self.params['EndTime']
#         else:
#             end_time = self.params['DownTime']
#         clipped_data = signal.data[(start_time <= signal.time) & (signal.time <= end_time)]
#         clipped_prop_dict = deepcopy(signal.attributes)
#         clipped_prop_dict['StartTime'] = start_time
#         clipped_prop_dict['SampleRate'] = (len(clipped_data) - 1) / (end_time - start_time)
#
#         return Signal(data=clipped_data, attributes=clipped_prop_dict)

class TimeTrimProcessor(BaseProcessor):
    """
        Given the start and end times, output the clipped timing signal
    """
    def __init__(self, start_time: float, end_time: float = None, end_time_tag: str = None):
        super().__init__()
        self._start_time = start_time
        self._end_time_tag = end_time_tag
        self._end_time = end_time

    def transform(self, signal: Signal) -> Signal:
        if self._end_time_tag:
            self._end_time = self.params[self._end_time_tag]
        if self._end_time is None:
            self._end_time = signal.time[-1]
        clipped_data = signal.data[(self._start_time <= signal.time) & (signal.time <= self._end_time)]
        clipped_prop_dict = deepcopy(signal.attributes)
        start_time_idx = np.argmax(signal.time > self._start_time)
        clipped_prop_dict['StartTime'] = signal.time[start_time_idx]

        return Signal(data=clipped_data, attributes=clipped_prop_dict)


class SliceProcessor(BaseProcessor):
    """
            input the point number of the window  and overlap rate of the given window ,
        then the sample rate is recalculated,  return a signal of time window sequence
    """
    def __init__(self, window_length: int, overlap: float):
        super().__init__()
        assert (0 <= overlap <= 1), "Overlap is not between 0 and 1."
        self.params.update({"WindowLength": window_length,
                            "Overlap": overlap})

    def transform(self, signal: Signal) -> Signal:
        window_length = self.params["WindowLength"]
        overlap = self.params["Overlap"]
        new_signal = deepcopy(signal)
        raw_sample_rate = new_signal.attributes["SampleRate"]
        step = round(window_length * (1 - overlap))

        down_time = new_signal.time[-1]

        down_time = round(down_time, 3)

        idx = len(signal.data)
        window = list()
        while (idx - window_length) >= 0:
            window.append(new_signal.data[idx - window_length:idx])
            idx -= step
        window.reverse()
        new_signal.attributes['SampleRate'] = raw_sample_rate * len(window) / (len(new_signal.data) - window_length + 1)
        new_signal.data = np.array(window)
        new_start_time = down_time - len(window) / new_signal.attributes['SampleRate']
        new_signal.attributes['StartTime'] = round(new_start_time, 3)
        return new_signal


class FFTProcessor(BaseProcessor):
    """
        processing signal by Fast Fourier Transform , return the maximum amplitude and the corresponding frequency
    """
    def __init__(self):
        super().__init__()

        self.amp_signal = None
        self.signal_rate = None
        self.fre_signal = None

    def transform(self, signal1: Signal, signal_rate: Signal):

        self.amp_signal = deepcopy(signal1)
        self.signal_rate = deepcopy(signal_rate)
        self.fre_signal = deepcopy(signal1)
        self.fft()
        self.amp_max()

        return self.amp_signal, self.fre_signal

    def fft(self):
        if self.amp_signal.data.ndim == 1:
            N = len(self.amp_signal.data)
            fft_y = fft(self.amp_signal.data)
            abs_y = np.abs(fft_y)
            normed_abs_y = abs_y / (N / 2)
            self.amp_signal.data = normed_abs_y[:int(N / 2)]
        elif self.amp_signal.data.ndim == 2:
            N = self.amp_signal.data.shape[1]
            R = self.amp_signal.data.shape[0]
            raw_cover = np.empty(shape=[0, int(N / 2)], dtype=float)
            for i in range(R):
                fft_y = fft(self.amp_signal.data[i])
                abs_y = np.abs(fft_y)
                normed_abs_y = abs_y / (N / 2)
                raw_cover = np.append(raw_cover, [normed_abs_y[:int(N / 2)]], axis=0)
            self.amp_signal.data = raw_cover

    def amp_max(self):
        fs = self.signal_rate.attributes['SampleRate']
        raw = self.amp_signal.data
        amp_cover = np.empty(shape=0, dtype=float)
        fre_cover = np.empty(shape=0, dtype=float)
        N = (raw.shape[1]) * 2
        f = (np.linspace(start=0, stop=N - 1, num=N) / N) * fs
        f = f[:int(N / 2)]
        for j in range(raw.shape[0]):
            list_max = (raw[j, :]).tolist()
            raw_max = max(list_max)
            max_index = list_max.index(max(list_max))
            f_rawmax = f[max_index]
            amp_cover = np.append(amp_cover, raw_max)
            fre_cover = np.append(fre_cover, f_rawmax)
        self.amp_signal.data = amp_cover
        self.fre_signal.data = fre_cover


class Mean(BaseProcessor):
    """
         Given a set of input signals, average each instant
    """
    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        new_signal = Signal(np.row_stack([sign.data for sign in signal.__iter__()]).T, signal.__getitem__(0).attributes)
        new_signal.data = np.mean(np.array(new_signal.data, dtype=np.float32), axis=1)
        return new_signal


class Concatenate(BaseProcessor):
    """
        calculate the mean and standard deviation of the given signal
    """
    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        new_signal = Signal(np.concatenate([sign.data for sign in signal.__iter__()], axis=0),
                            signal.__getitem__(0).attributes)

        return new_signal

# class AlarmTag(BaseProcessor):
#     """
#             Give arbitrary signals, extract downtime, timeline,
#         and generate actual warning time labels
#
#     """
#
#     def __init__(self, lead_time, fs, start_time):
#         super().__init__()
#         self.lead_time = lead_time
#         self.fs = fs
#         self.start_time = start_time
#
#     def transform(self, signal: Signal):
#         copy_signal = deepcopy(signal)
#         undisrupt_number = int(self.fs * (self.params['DownTime'] + self.lead_time))
#         new_signal = Signal(data=np.empty(shape=[0], dtype=bool), attributes=dict())
#         new_data = np.array([bool(0) for i in range(undisrupt_number)])
#         new_data = np.append(new_data,
#                              np.array([bool(1) for i in range(int(self.fs * copy_signal.time[-1] - undisrupt_number))]),
#                              axis=0)
#         new_signal.attributes['SampleRate'] = self.fs
#         new_signal.attributes['StartTime'] = self.start_time
#         new_signal.data = new_data
#         return new_signal


class AlarmTag(BaseProcessor):
    """
            Give arbitrary signals, extract downtime, timeline,
        and generate actual warning time labels

    """
    def __init__(self, lead_time, disruption_label: str, downtime_label: str):
        super().__init__()
        self.lead_time = lead_time
        self.disruption_label =disruption_label
        self.downtime_label = downtime_label

    def transform(self, signal: Signal):
        copy_signal = deepcopy(signal)
        fs = copy_signal.attributes['SampleRate']
        start_time = copy_signal.attributes['StartTime']
        if self.params['IsDisrupt'] == 1:
            undisrupt_number = int(fs * (self.params[self.downtime_label] + self.lead_time))
            new_data = np.zeros(shape=undisrupt_number, dtype=int)
            new_data = np.append(new_data, np.ones(shape=len(copy_signal.data)-undisrupt_number), axis=0)
        else:
            new_data = np.zeros(shape=len(copy_signal.data), dtype=int)

        new_signal = Signal(data=new_data, attributes=dict())
        new_signal.attributes['SampleRate'] = fs
        new_signal.attributes['StartTime'] = start_time

        return new_signal





