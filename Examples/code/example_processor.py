import math

from jddb.processor import Signal, Shot, ShotSet, BaseProcessor
from typing import Optional, List
from scipy.fftpack import fft
from copy import deepcopy
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from scipy import signal as sig
from copy import deepcopy


class TimeClipProcessor(BaseProcessor):
    def __init__(self, start_time, end_time=None):
        super().__init__()
        self.params.update({
            "StartTime": start_time,
            "EndTime": end_time
        })

    def transform(self, signal: Signal) -> Signal:
        start_time = self.params['StartTime']
        if self.params['EndTime']:
            end_time = self.params['EndTime']
        else:
            end_time = self.params['DownTime']
        clipped_data = signal.data[(start_time <= signal.time) & (signal.time <= end_time)]
        clipped_prop_dict = deepcopy(signal.attributes)
        clipped_prop_dict['StartTime'] = start_time
        clipped_prop_dict['SampleRate'] = (len(clipped_data) - 1) / (end_time - start_time)

        return Signal(data=clipped_data, attributes=clipped_prop_dict)


class SliceProcessor(BaseProcessor):  # 时长+重叠率
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
    def __init__(self):
        super().__init__()

    def transform(self, signal: Signal):
        new_signal = deepcopy(signal)
        if new_signal.data.ndim == 1:
            N = len(new_signal.data)
            fft_y = fft(new_signal.data)
            abs_y = np.abs(fft_y)
            normed_abs_y = abs_y / (N / 2)
            new_signal.data = normed_abs_y[:int(N / 2)]
        elif new_signal.data.ndim == 2:
            N = new_signal.data.shape[1]
            R = new_signal.data.shape[0]
            raw_cover = np.empty(shape=[0, int(N / 2)], dtype=float)
            for i in range(R):
                fft_y = fft(new_signal.data[i])
                abs_y = np.abs(fft_y)
                normed_abs_y = abs_y / (N / 2)
                raw_cover = np.append(raw_cover, [normed_abs_y[:int(N / 2)]], axis=0)
            new_signal.data = raw_cover
        return new_signal


class AmpMaxProcessor(BaseProcessor):
    def __init__(self, fs):
        super().__init__()
        self.fs = fs

    def transform(self, signal: Signal):
        new_signal = deepcopy(signal)
        raw = new_signal.data
        raw_cover = np.empty(shape=[0, 2], dtype=float)
        N = (raw.shape[1]) * 2
        f = np.linspace(start=0, stop=N - 1, num=N) / N * self.fs
        f = f[:int(N / 2)]
        for j in range(raw.shape[0]):
            list_max = (raw[j, :]).tolist()
            raw_max = max(list_max)
            max_index = list_max.index(max(list_max))
            f_rawmax = f[max_index]
            raw_cover = np.append(raw_cover, [[raw_max, f_rawmax]], axis=0)
            new_signal.data = raw_cover
        return new_signal


class BandPassFilterProcessor(BaseProcessor):
    """
    -------
    带通滤波：
    high_fre:低通截止频率
    high_pass:高通截止频率
    oder:滤波器阶数
    注意:
    0< wn =(2 * self.high_fre / fs) <1
    0< wn =(2 * self.high_pass / fs) <1
    -------
    """

    def __init__(self, low_pass, high_pass, oder):
        super().__init__()
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.oder = int(oder)

    def transform(self, signal: Signal):
        new_signal = deepcopy(signal)
        fs = new_signal.attributes["SampleRate"]
        b, a = sig.butter(self.oder, [2 * self.low_pass / fs, 2 * self.high_pass / fs], btype='bandpass')
        new_signal.data = sig.filtfilt(b, a, new_signal.data)  # new_signal.raw为要过滤的信号
        return new_signal


class M_mode(BaseProcessor):
    """
           -------
           用于计算模数
           data1，data2为两道Mirnov探针信号（尽可能近）
           chip_time为做互功率谱时的窗口时间长度，默认为5ms
           down_number为做互功率谱时的降点数（down_number = 1，即取了全时间窗的点，即FFT窗仅为一个），默认为8
           low_fre为所需最低频率，默认为2kHz
           high_fre为所需最高频率，默认为100kHz
           step_fre为选取最大频率时的步长，默认为3kHz
           max_number为选取最大频率的个数，默认为3个
           var_th为频率间的方差阈值，默认为1e-13
           real_angle为两道Mirnov探针间的极向空间角度差，默认为15°
           coherence_th为互相关系数阈值，默认为0.95
           -------
    """

    def __init__(self, real_angle=7.5, chip_time=5e-3, down_number=8, low_fre=1e2, high_fre=2e4, step_fre=3e3,
                 max_number=int(3), var_th=1e-12, coherence_th=0.9):
        # var_th=1e-13, coherence_th=0.95
        super().__init__()
        self.chip_time = chip_time
        self.down_number = down_number
        self.low_fre = low_fre
        self.high_fre = high_fre
        self.step_fre = step_fre
        self.max_number = max_number
        self.var_th = var_th
        self.real_angle = real_angle
        self.coherence_th = coherence_th

    # def band_pass_filter(self, data, sampling_freq, high_fre, low_fre):
    #
    #     # ba1 = sig.butter(4, (2 * high_fre / sampling_freq), "lowpass")
    #     # filter_data1 = sig.filtfilt(ba1[0], ba1[1], data)
    #     # ba2 = sig.butter(4, (2 * low_fre / sampling_freq), "highpass")
    #     # filter_data = sig.filtfilt(ba2[0], ba2[1], filter_data1)
    #     b, a = sig.butter(4, [(2 * low_fre) / sampling_freq, (2 * high_fre) / sampling_freq], btype='bandpass')
    #     filter_data = sig.filtfilt(b, a, data)  # new_signal.raw为要过滤的信号
    #     return filter_data

    def transform(self, signal1: Signal, signal2: Signal):
        signal1 = deepcopy(signal1)
        signal2 = deepcopy(signal2)
        number_of_a_window = int(self.chip_time * signal1.attributes['SampleRate'])  # window length
        number_of_windows = len(signal1.time) // number_of_a_window  # number of the chip in the window
        number_of_f_low = int(self.low_fre * self.chip_time)  # lowest frequency
        number_of_f_high = int(self.high_fre * self.chip_time)  # Highest frequency
        number_of_f_step = int(self.step_fre * self.chip_time)  # select max_frequency length
        new_signal1 = Signal(data=np.empty(shape=[0, 5], dtype=float), attributes=dict())
        new_signal2 = Signal(data=np.empty(shape=[0, 5], dtype=float), attributes=dict())
        new_signal3 = Signal(data=np.empty(shape=[0, 5], dtype=float), attributes=dict())

        # signal1.prop_dict['SampleRate']为原采样率
        new_signal1.attributes['SampleRate'] = signal1.attributes['SampleRate'] / number_of_a_window
        new_signal2.attributes['SampleRate'] = signal1.attributes['SampleRate'] / number_of_a_window
        new_signal3.attributes['SampleRate'] = signal1.attributes['SampleRate'] / number_of_a_window
        new_signal1.attributes['StartTime'] = signal1.attributes['StartTime'] + self.chip_time
        new_signal2.attributes['StartTime'] = signal1.attributes['StartTime'] + self.chip_time
        new_signal3.attributes['StartTime'] = signal1.attributes['StartTime'] + self.chip_time
        # filter
        signal1 = BandPassFilterProcessor(low_pass=self.low_fre, high_pass=self.high_fre, oder=4).transform(signal1)
        signal2 = BandPassFilterProcessor(low_pass=self.low_fre, high_pass=self.high_fre, oder=4).transform(signal2)
        # slide
        for i in range(number_of_windows):
            chip_data1 = signal1.data[
                         int(number_of_a_window * i):int(number_of_a_window * i + number_of_a_window)] \
                         - np.mean(
                signal1.data[int(number_of_a_window * i):int(number_of_a_window * i + number_of_a_window)])
            chip_data2 = signal2.data[
                         int(number_of_a_window * i):int(number_of_a_window * i + number_of_a_window)] \
                         - np.mean(
                signal2.data[int(number_of_a_window * i):int(number_of_a_window * i + number_of_a_window)])
            """做互功率谱,看相关性,并取互功率谱幅值、相位"""
            # calculate cross spectral density
            (f, csd) = sig.csd(chip_data1, chip_data2, fs=signal1.attributes['SampleRate'], window='hann',
                               nperseg=number_of_a_window // self.down_number, scaling='density')
            (f_coherence, coherence) = sig.coherence(chip_data1, chip_data2, fs=signal1.attributes['SampleRate'],
                                                     window='hann',
                                                     nperseg=number_of_a_window // self.down_number)
            abs_csd = np.abs(csd)
            # log_abs_csd = 20 * np.log(abs_csd)
            log_abs_csd = 1e6 * abs_csd
            phase_csd = np.angle(csd) * 180 / np.pi
            """在信号相关系数阈值之上的angle存入数组，否则该数组为空"""
            angle_csd = np.where(coherence > self.coherence_th, phase_csd, np.nan)
            """csd中选取一段频率，低频到高频，且做方差"""
            # abs_csd_chosen = np.abs(csd[number_of_f_low // down_number: number_of_f_high // down_number])
            abs_csd_chosen = np.abs(csd[number_of_f_low // self.down_number: number_of_f_high // self.down_number])
            var_csd = np.var(np.abs(abs_csd_chosen))
            # 求互功率谱均值，若最大频率大于最小频率，且小于ch_th就证明存在明显的模式, 目前使用方差与互相关系数做约束
            # 判断是否有明显模式
            if var_csd < self.var_th:
                new_signal1.data = np.concatenate((new_signal1.data, np.zeros((1, 5), dtype=np.float)), axis=0)
                new_signal2.data = np.concatenate((new_signal2.data, np.zeros((1, 5), dtype=np.float)), axis=0)
                new_signal3.data = np.concatenate((new_signal3.data, np.zeros((1, 5), dtype=np.float)), axis=0)
            else:
                down_number_of_step = number_of_f_step // self.down_number
                overall_index_of_max_csd_in_every_piece = []
                max_csd_in_every_piece = []
                """csd分片段处理,滑动窗口每段找出一个最大值,并存其全局最大值索引"""
                for j in range(len(abs_csd_chosen) // down_number_of_step):  # 把csd分几个片段处理
                    abs_csd_piece = np.abs(abs_csd_chosen[j * down_number_of_step:(j + 1) * down_number_of_step])
                    max_csd_in_every_piece.append(max(abs_csd_piece))  # 滑动窗口每段找出一个最大值
                    # 全局最大值索引,max_csd_in_every_piece索引与overall_index_of_max_csd_in_every_piece索引一一对应
                    overall_index_of_max_csd_in_every_piece.append(
                        abs_csd_piece.argsort()[::-1][
                            0] + number_of_f_low // self.down_number + j * down_number_of_step)
                max_csd_in_every_piece = np.array(max_csd_in_every_piece)
                # print("tmp.argsort()[::-1][0:max_number]:")#由小到大的数据的索引
                # "tmp.argsort()由大到小的数据的索引
                """取所有片段的最大的csd中，前三个最大值的局部索引"""
                local_index_of_top_three_max = max_csd_in_every_piece.argsort()[::-1][
                                               0:self.max_number]  # 取所有片段的最大的csd中，前三个最大值的局部索引
                """将前三个最大值局部索引转为全局索引"""
                # local_index_of_top_three_max值与max_csd_in_every_piece索引一一对应
                overall_index_of_top_three_max = np.empty(shape=[0], dtype=float)
                for local_index in local_index_of_top_three_max:
                    overall_index_of_top_three_max = np.append(overall_index_of_top_three_max,
                                                               overall_index_of_max_csd_in_every_piece[local_index])
                overall_index_of_top_three_max = overall_index_of_top_three_max.astype(int)
                """将top three的频率、幅值、相位对应取出"""
                """ n*3 m1(m, 幅值, 频率, 相位),m2(),m3()"""

                new_signal1_add = np.array([angle_csd[overall_index_of_top_three_max[0]] / self.real_angle,
                                            1000 * log_abs_csd[overall_index_of_top_three_max[0]] / f[
                                                overall_index_of_top_three_max[0]],
                                            log_abs_csd[overall_index_of_top_three_max[0]],
                                            f[overall_index_of_top_three_max[0]],
                                            angle_csd[overall_index_of_top_three_max[0]],
                                            ]).reshape(1, 5)
                new_signal2_add = np.array([angle_csd[overall_index_of_top_three_max[1]] / self.real_angle,
                                            1000 * log_abs_csd[overall_index_of_top_three_max[1]] / f[
                                                overall_index_of_top_three_max[1]],
                                            log_abs_csd[overall_index_of_top_three_max[1]],
                                            f[overall_index_of_top_three_max[1]],
                                            angle_csd[overall_index_of_top_three_max[1]]
                                            ]).reshape(1, 5)
                new_signal3_add = np.array([angle_csd[overall_index_of_top_three_max[2]] / self.real_angle,
                                            1000 * log_abs_csd[overall_index_of_top_three_max[2]] / f[
                                                overall_index_of_top_three_max[2]],
                                            log_abs_csd[overall_index_of_top_three_max[2]],
                                            f[overall_index_of_top_three_max[2]],
                                            angle_csd[overall_index_of_top_three_max[2]]]).reshape(1, 5)
                if len(new_signal1_add) == 0:
                    new_signal1_add = np.zeros(shape=[1, 5], dtype=float)
                new_signal1.data = np.append(new_signal1.data, new_signal1_add, axis=0)

                if len(new_signal2_add) == 0:
                    new_signal2_add = np.zeros(shape=[1, 5], dtype=float)
                new_signal2.data = np.append(new_signal2.data, new_signal2_add, axis=0)

                if len(new_signal3_add) == 0:
                    new_signal3_add = np.zeros(shape=[1, 5], dtype=float)
                new_signal3.data = np.append(new_signal3.data, new_signal3_add, axis=0)
        return new_signal1, new_signal2, new_signal3


class Mode_to_int(BaseProcessor):
    """
    将模数变成整数，
    delta = m-int(m)
    [0,0.4] m=int(m)
    [0.4,0.6] m=int(m)+0.5
    [0.6,1）  m=int(m)+1
    """

    def __init__(self):
        super().__init__()

    def transform(self, signal: Signal):
        new_signal = deepcopy(signal)
        new_signal.data[:, 0] = np.abs(new_signal.data[:, 0])
        delta_signal = new_signal.data[:, 0] - np.trunc(new_signal.data[:, 0])
        int_signal_raw = np.trunc(new_signal.data[:, 0])
        for i in range(new_signal.data.shape[0]):
            # x [0,0.4] [0.4,0.6] [0.6,1）
            if delta_signal[i] > 0.6:
                int_signal_raw[i] += 1
            elif delta_signal[i] >= 0.4 and delta_signal[i] >= 0.6:
                int_signal_raw[i] += 0.5
        new_signal.data[:, 0] = int_signal_raw
        return new_signal


class ModeClassification(BaseProcessor):
    """
       -------
       judge whether Br(n=number) existing by using 3 signals
       -------
       """

    def __init__(self, m_munber):
        super().__init__()
        self.m_munber = m_munber

    def transform(self, signal_most: Signal, signal_sec: Signal, signal_last: Signal, signal_lock: Signal):
        signal_most = deepcopy(signal_most)
        signal_sec = deepcopy(signal_sec)
        signal_last = deepcopy(signal_last)
        signal_lock = deepcopy(signal_lock)
        new_signal_m = Signal(data=np.empty(shape=[0], dtype=bool), attributes=dict())
        new_signal_b_amp = Signal(data=np.empty(shape=[0], dtype=float), attributes=dict())
        new_signal_fre = Signal(data=np.empty(shape=[0], dtype=float), attributes=dict())
        # signal_most.prop_dict['SampleRate']为原采样率
        new_signal_m.attributes['SampleRate'] = signal_most.attributes['SampleRate']
        new_signal_m.attributes['StartTime'] = signal_most.attributes['StartTime']
        new_signal_b_amp.attributes['SampleRate'] = signal_most.attributes['SampleRate']
        new_signal_b_amp.attributes['StartTime'] = signal_most.attributes['StartTime']
        new_signal_fre.attributes['SampleRate'] = signal_most.attributes['SampleRate']
        new_signal_fre.attributes['StartTime'] = signal_most.attributes['StartTime']
        for i in range(len(signal_most.data[:, 0])):
            m = [signal_most.data[i, 0], signal_sec.data[i, 0], signal_last.data[i, 0]]
            b_amp = [signal_most.data[i, 1], signal_sec.data[i, 1], signal_last.data[i, 1]]
            b_fre = [signal_most.data[i, 2], signal_sec.data[i, 2], signal_last.data[i, 2]]
            number_index = [index for index, number in enumerate(m) if number == self.m_munber]
            if len(number_index) != 0:
                tag = bool(1)
                if signal_lock.data[i] == bool(1):
                    max_fre = np.array([0])
                    max_amp = [np.nan]
                else:
                    amp = [b_amp[index] for index, number in enumerate(m) if number == self.m_munber]
                    fre = [b_fre[index] for index, number in enumerate(m) if number == self.m_munber]
                    max_amp = max(amp)  # 求列表最大值
                    max_idx = amp.index(max_amp)  # 求最大值对应索引
                    max_fre = np.array([fre[max_idx]])
                    max_amp = np.array([max_amp])
            else:
                tag = bool(0)
                max_amp = [np.nan]
                max_fre = [np.nan]
            new_signal_m.data = np.append(new_signal_m.data, np.array([tag]), axis=0)
            new_signal_b_amp.data = np.append(new_signal_b_amp.data, max_amp, axis=0)
            new_signal_fre.data = np.append(new_signal_fre.data, max_fre, axis=0)

        return new_signal_m, new_signal_b_amp, new_signal_fre


class LockModeJudege(BaseProcessor):
    def __init__(self):
        super().__init__()

    def transform(self, signal_most: Signal, signal_sec: Signal, signal_last: Signal, amp_fre_signal: Signal):
        signal_most = deepcopy(signal_most)
        signal_sec = deepcopy(signal_sec)
        signal_last = deepcopy(signal_last)
        amp_fre_signal = deepcopy(amp_fre_signal)
        new_signal_lock = Signal(data=np.empty(shape=[0], dtype=bool), attributes=dict())
        new_signal_lock.attributes['SampleRate'] = signal_most.attributes['SampleRate']
        new_signal_lock.attributes['StartTime'] = signal_most.attributes['StartTime']
        lock_data = np.empty(shape=[0], dtype=bool)
        for i in range(len(signal_most.data[:, 0])):
            if amp_fre_signal.data[i, 1] < 900 or (
                    signal_most.data[i, 1] == float("inf") or signal_sec.data[i, 1] == float("inf") or
                    signal_last.data[
                        i, 1] == float("inf")):
                lock_data = np.append(lock_data, np.array([bool(1)]), axis=0)
            else:
                lock_data = np.append(lock_data, np.array([bool(0)]), axis=0)

        new_signal_lock.data = lock_data
        return new_signal_lock


class N1mode(BaseProcessor):
    """
       -------
       compute the amp, phi of Br(n=1) by using 4 exsad signals
       -------
    """

    def __init__(self):
        super().__init__()

    def transform(self, exsad0: Signal, exsad1: Signal, exsad2: Signal, exsad3: Signal):
        new_signal = deepcopy(exsad0)
        raw = [exsad0.data, exsad1.data, exsad2.data, exsad3.data]
        amp, phase = self.locked_mode(exsad2.time, vbr0=raw, theta=None)
        exsad_amp_fre = [np.array([amp[0], phase[0]])]
        for j in range(len(exsad0.data) - 1):
            exsad_amp_fre_vstack = [np.array([amp[j + 1], phase[j + 1]])]
            exsad_amp_fre = np.vstack((exsad_amp_fre, exsad_amp_fre_vstack))
        new_signal.data = exsad_amp_fre
        return new_signal

    def n_1_mode(self, theta, br, deg):
        """
        -------
        compute br(n=1)
        theta: subtraction  of toroidal space angle of 2 exsad
        br = amp*cos(theta+phase)
        deg:
        -------
        """
        theta1 = theta[0] / 180 * math.pi
        theta2 = theta[1] / 180 * math.pi
        D = math.sin(theta1 - theta2)
        br1 = br[0]
        br2 = br[1]
        amp = (br1 ** 2 + br2 ** 2 - 2 * br1 * br2 * math.cos(theta1 - theta2)) ** 0.5 / abs(math.sin(theta1 - theta2))
        cos_phi = (-br2 * math.cos(theta1) + br1 * math.cos(theta2)) / D
        sin_phi = (br2 * math.sin(theta1) - br1 * math.sin(theta2)) / D
        tanPhi = sin_phi / cos_phi
        # phase of origin is -(phs + 2 * pi * f * t)
        # phase of b ^ max is pi / 2 - (phs + 2 * pi * f * t)
        # the variable in sine function
        dlt0 = np.zeros(len(tanPhi), dtype=np.float)
        for i in range(len(tanPhi)):
            dlt0[i] = math.atan(tanPhi[i]) / math.pi * 180 + 180 * np.floor((1 - np.sign(cos_phi[i])) / 2) - 90
        # the variable in cosine function, so it is also the phase of b_theta maximum.
        phase = self.deg_2deg(-dlt0, deg)
        # the phase of b ^ max
        return amp, phase

    def locked_mode(self, time, vbr0, theta=None):
        """
        -------
        shot: shot_number which effecting NSbr
        time: timeline
        vbr0: 4 * len(time) array , time sequence of four exsad signals
        theta: subtraction  of toroidal space angle of 2 exsad
        -------
        """
        if theta is None:
            theta = [67.5, 157.5]
        if (self.params["Shot"] - 1061320) > 0:
            NSbr = [2.6613, 5.968, 2.0540, 2.49]
            tau_br = [10e-3, 100e-3, 10e-3, 10e-3]
        elif (self.params["Shot"] - 1052000) * (self.params["Shot"] - 1052672) < 0 or self.params["Shot"] > 1052900:
            NSbr = [2.37, 5.32, 1.84, 2.49]
            tau_br = [10e-3, 100e-3, 10e-3, 10e-3]
        else:
            NSbr = [1.09, -2.66, -1.14, -2.49]
            tau_br = [10e-3, 10e-3, 10e-3, 10e-3]
        br_Saddle = np.zeros((4, len(time)), dtype=np.float)
        for j1 in range(len(NSbr)):
            br_Saddle[j1] = vbr0[j1] / NSbr[j1] * tau_br[j1] * 1e4
        br_odd = np.zeros((2, len(time)), dtype=np.float)  # 创建2维数组存放诊断数据

        br_odd[0] = br_Saddle[0] - br_Saddle[2]
        br_odd[1] = br_Saddle[1] - br_Saddle[3]
        amp, phase = self.n_1_mode(theta, br_odd, 180)
        return amp, phase

    def deg_2deg(self, deg0, deg):
        """
        -------
        deg: change degree from (0, - 360) to (0+deg, -360+deg)
        -------
        """
        deg1 = deg0 * 0
        mm = np.size(deg0)
        for mm1 in range(1, mm):
            deg1[mm1] = deg0[mm1] - math.floor((deg0[mm1] + (360 - deg)) / 360) * 360
        return deg1

# a = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1]
# m = []
# for i, v in enumerate(a):
#     if i < len(m) - 1:
#         continue
#     if i < (len(a) - 3):
#         if a[i] == 1 and a[i + 1] == 1 and a[i + 2] == 1:
#             for j in range(3):
#                 m.append([1])
#         else:
#             m.append([0])





