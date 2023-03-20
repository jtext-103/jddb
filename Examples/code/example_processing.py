from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import AutoMinorLocator

from jddb.processor import ShotSet, Shot, Signal, BaseProcessor
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor
from jddb.file_repo import FileRepo
import matplotlib
from matplotlib import pyplot as plt
from example_processor import *
import pandas as pd
import numpy as np
import math
import warnings
import h5py
import os
import json


def read_config():
    """"读取配置"""
    with open("config_mode_prediction.json", 'r') as json_file:
        config = json.load(json_file, strict=False)
    return config


config = read_config()
input_signals = config["windows"]["diagnosis"]["input_signals"]
slice_signal = config["windows"]["diagnosis"]["slice_signal"]
MA_m = config["windows"]["diagnosis"]["MA_m"]
exsad_n = config["windows"]["diagnosis"]["exsad_n"]
m_mode_number = config["windows"]["diagnosis"]["m_mode_number"]
amp_fre_signal = config["windows"]["diagnosis"]["amp_fre_signal"]
resampled_rate = config["windows"]["resampled_rate"]
clip_second = config["windows"]["clip_second"]
low_pass = config["windows"]["low_pass"]
oder = config["windows"]["oder"]
real_angle = config["windows"]["real_angle"]
high_pass = config["windows"]["high_pass"]
window_length = config["windows"]["window_length"]
overlap_5ms = config["windows"]["overlap_5ms"]
overlap_tenms = config["windows"]["overlap_tenms"]
raw_shotsets_path = config["windows"]["path"]["raw_shotsets_path"]
to_be_processed_path = config["windows"]["path"]["to_be_processed_shotsets_path"]
resampled_shotsets_path = config["windows"]["path"]["resampled_shotsets_path"]
fre_picture_path = config["windows"]["path"]["fre_picture_path"]

# load data
# base_path = os.path.join(raw_shotsets_path, "$shot_2$XX", "$shot_1$X")
# file_repo = FileRepo(base_path)
# n_mode_dataset = ShotSet(file_repo)
resampled_shotsets_path = os.path.join(resampled_shotsets_path, "$shot_2$XX", "$shot_1$X")
n_mode_dataset = ShotSet(FileRepo(resampled_shotsets_path))
shotlist = n_mode_dataset.shot_list


# # n_mode_dataset = ShotSet(FileRepo(r'G:\datapractice\example_lry\raw_shotsets\$shot_2$XX\$shot_1$X'))
# # n_mode_dataset = ShotSet(FileRepo(raw_shotsets_path))
# shotlist = n_mode_dataset.shot_list
# shot = n_mode_dataset.get_shot(shotlist[0])
# #drop siganl not
# n_mode_dataset = n_mode_dataset.remove(tags=MA_m, keep=True,
#                                             save_repo=FileRepo(r'G:\datapractice\example_lry\Example_shot_files\$shot_2$XX\$shot_1$X'))
# clip
# n_mode_dataset = ShotSet(FileRepo(r'G:\datapractice\example_lry\Example_shot_files\$shot_2$XX\$shot_1$X'))
# n_mode_dataset = n_mode_dataset.process(processor=TimeClipProcessor(start_time=0),
#                                         input_tags=exsad_n + MA_m,
#                                         output_tags=exsad_n + MA_m,
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path))
# # bandpass MA exsad
# n_mode_dataset = n_mode_dataset.process(processor=BandPassFilterProcessor(low_pass=low_pass, high_pass=high_pass,
#                                                                           oder=oder),
#                                         input_tags=exsad_n + MA_m,
#                                         output_tags=exsad_n + MA_m,
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path))  # save_repo=FileRepo(resampled_shotsets_path))
# # resample exsad MA
# n_mode_dataset = n_mode_dataset.process(processor=ResamplingProcessor(resampled_rate),
#                                         input_tags=exsad_n + MA_m,
#                                         output_tags=exsad_n + MA_m,
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path))
# slice MA
# n_mode_dataset = n_mode_dataset.process(processor=SliceProcessor(window_length=window_length, overlap=0),
#                                         input_tags=[MA_m[0]],
#                                         output_tags=[slice_signal[0]],
#                                         save_repo=FileRepo(
#                                                 resampled_shotsets_path))
# # fft MA
# n_mode_dataset = n_mode_dataset.process(processor=FFTProcessor(),
#                                         input_tags=[slice_signal[0]],
#                                         output_tags=[slice_signal[0]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path))
# # fre_amp MA
# n_mode_dataset = n_mode_dataset.process(processor=AmpMaxProcessor(resampled_rate),
#                                         input_tags=[slice_signal[0]],
#                                         output_tags=[amp_fre_signal[0]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path))
# n_mode_dataset = ShotSet(FileRepo(resampled_shotsets_path))
# shotlist = n_mode_dataset.shot_list
# signal_to_fft_shotsets_path = os.path.join(signal_to_fft_shotsets_path, "$shot_2$XX", "$shot_1$X")
# n_mode_dataset = n_mode_dataset.process(processor=N1mode(),
#                                         input_tags=[exsad_n],
#                                         output_tags=[amp_fre_signal[4]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path))
# def spectrogram_tm(shot_no, dataset, picture_path):
#     # p1 spectroram p2 ip
#     # p3 MA         p4 exsad
#     # p5 MA_amp_fre p6 exsad_n=1
#     matplotlib.use('Agg')  # 以下绘制的图形不会在窗口显示
#     shot_plt = dataset.get_shot(shot_no)
#     signal_ip = shot_plt.get("\\ip")  # p2
#     signal_MA_POL2_P01 = shot_plt.get(MA_m[0])  # p3
#     signal_exsad_n = shot_plt.get(exsad_n[0])  # p4 exsad1
#     signal_MA_amp_fre = shot_plt.get(amp_fre_signal[0])  # p5
#     signal_exsad_amp_phi = shot_plt.get(amp_fre_signal[4])  # p6
#     end_time = max([shot_plt.labels["DownTime"], signal_MA_POL2_P01.time[-1], signal_exsad_n.time[-1]])  #
#     # end_time = signal_MA_POL2_P01.time[-1]
#     # 绘制图形并存入指定路径
#     # 图片保存路径
#     dir_name = os.path.join(picture_path, str(shot_no) + '.jpg')
#     fig = plt.figure(1, figsize=(23, 13))
#     for j in range(6):
#         plt.subplots_adjust(wspace=0.4, hspace=0.2)
#         if j == 0:
#             # "\MA_POL_CA01T"
#             ax = plt.subplot(3, 2, j + 1)  # p1
#             plt.ylabel("MA_POL2_P01")
#             # plt.xticks(np.arange(0.1, end_time, 0.05))
#             # ax.xaxis.set_major_locator(MultipleLocator(1.0))
#             # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
#             # plt.xlim([0, end_time])
#
#             # plt.specgram(np.concatenate((np.zeros(int(clip_second*signal_MA_POL2_P01.attributes['SampleRate'])),
#             #                              signal_MA_POL2_P01.data),axis=0),
#             #                                 Fs=signal_MA_POL2_P01.attributes['SampleRate'], cmap='hsv')
#             # signal_MA_POL2_P01.data[signal_MA_POL2_P01.attributes['SampleRate']*signal_MA_POL2_P01.attributes['StartTime']:-1]
#             plt.specgram(signal_MA_POL2_P01.data,
#                          Fs=signal_MA_POL2_P01.attributes['SampleRate'],
#                          cmap='plasma')  #
#             # pxx, freq, t, cax =
#             plt.colorbar(cax=fig.add_axes([0.05, 0.4, 0.01, 0.5]))
#         if j == 2:
#             plt.subplot(3, 2, j + 1)  # 图3
#             # "MA"
#             plt.xlim([0, end_time])
#             plt.ylabel("MA_POL2_P01")
#             plt.plot(signal_MA_POL2_P01.time, signal_MA_POL2_P01.data)
#         if j == 4:
#             plt.subplot(3, 2, j + 1)
#             plt.xlim([0, end_time])
#             plt.xlabel("time")
#             ax1 = plt.subplot(3, 2, j + 1)  # 图5
#             ax2 = ax1.twinx()  # 做镜像处理
#             ax1.plot(signal_MA_amp_fre.time, signal_MA_amp_fre.data[:, j - 4], 'b-')  # amp
#             ax2.plot(signal_MA_amp_fre.time, signal_MA_amp_fre.data[:, j - 3], 'r-')  # fre
#             ##ax1.set_xlabel('X data')  # 设置x轴标题
#             ax1.set_ylabel("Y_5ms_Max_Amp", color='b')  # 设置Y1轴标题
#             ax2.set_ylabel("Y_5ms_fre", color='r')  # 设置Y2轴标题
#         if j == 1:
#             # "ip"
#             plt.subplot(3, 2, j + 1)  # 图2
#             plt.xlim([0, signal_ip.time[-1] - 1])
#             plt.ylabel("ip")
#             plt.plot(signal_ip.time, signal_ip.data)
#         if j == 3:
#             plt.subplot(3, 2, j + 1)  # 图4
#             # "exsad1"
#             plt.xlim([0, end_time])
#             plt.ylabel("exsad1")
#             plt.plot(signal_exsad_n.time, signal_exsad_n.data)
#         if j == 5:
#             plt.subplot(3, 2, j + 1)  # 图6
#             # "exsad_amp_phi"
#             plt.xlabel("time")
#             plt.xlim([0, end_time])
#             ax1 = plt.subplot(3, 2, j + 1)
#             # ax2 = ax1.twinx()  # 做镜像处理
#             ax1.plot(signal_exsad_amp_phi.time, signal_exsad_amp_phi.data[:, j - 5], 'b-')
#             # ax2.plot(signal_exsad_amp_phi.time, signal_exsad_amp_phi.raw[:, j-4], 'r-')
#             ##ax1.set_xlabel('X data')  # 设置x轴标题
#             ax1.set_ylabel("exsad1_Amp", color='b')  # 设置Y1轴标题
#             # ax2.set_ylabel("exsad1_phi", color='r')  # 设置Y2轴标题
#         ax = plt.gca()
#         if j != 1:
#             ax.xaxis.set_major_locator(MultipleLocator(0.05))
#         else:
#             ax.xaxis.set_major_locator(MultipleLocator(0.1))
#         plt.grid(axis="x", linestyle='-.', which='major')
#     plt.suptitle(str(shot_no) + "_5ms", x=0.5, y=0.98)
#     plt.savefig(dir_name)
#     plt.close()
#
#
# for i in range(len(shotlist)):
#     spectrogram_tm(shotlist[i], n_mode_dataset, fre_picture_path)

"""
-------
3.18
1.根据图像选M_mode阈值
2.找出m=1 ,m=2, ...的标签
  lock_mode 时 fre=0，amp=Nan
3.写个down_time+t 的，disraptive——signal标签(sample-rate*****)
4.修改roc，修改report函数

-------
"""
# n_mode_dataset = n_mode_dataset.process(processor=M_mode(),
#                                         input_tags=[[MA_m[0], MA_m[1]]],
#                                         output_tags=[[amp_fre_signal[1], amp_fre_signal[2], amp_fre_signal[3]]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path))
# ## m to int
# n_mode_dataset = n_mode_dataset.process(processor=Mode_to_int(),
#                                         input_tags=[amp_fre_signal[1]],
#                                         output_tags=[m_mode_number[3]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path))
#
# n_mode_dataset = n_mode_dataset.process(processor=Mode_to_int(),
#                                         input_tags=[amp_fre_signal[2]],
#                                         output_tags=[m_mode_number[4]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path)
#                                         )
#
# n_mode_dataset = n_mode_dataset.process(processor=Mode_to_int(),
#                                         input_tags=[amp_fre_signal[3]],
#                                         output_tags=[m_mode_number[5]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path)
#                                         )
##lockmode
#
# n_mode_dataset = n_mode_dataset.process(processor=LockModeJudege(),
#                                         input_tags=[[m_mode_number[3], m_mode_number[4], m_mode_number[5], amp_fre_signal[0]]],
#                                         output_tags=[m_mode_number[6]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path)
#                                         )
# # m classification
# n_mode_dataset = n_mode_dataset.process(processor=ModeClassification(m_munber=1),
#                                         input_tags=[[m_mode_number[3], m_mode_number[4], m_mode_number[5], m_mode_number[6]]],
#                                         output_tags=[[m_mode_number[0], amp_fre_signal[5], amp_fre_signal[8]]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path)
#                                         )
# n_mode_dataset = n_mode_dataset.process(processor=ModeClassification(m_munber=2),
#                                         input_tags=[[m_mode_number[3], m_mode_number[4], m_mode_number[5], m_mode_number[6]]],
#                                         output_tags=[[m_mode_number[1], amp_fre_signal[6], amp_fre_signal[9]]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path)
#                                         )
# n_mode_dataset = n_mode_dataset.process(processor=ModeClassification(m_munber=3),
#                                         input_tags=[[m_mode_number[3], m_mode_number[4], m_mode_number[5], m_mode_number[6]]],
#                                         output_tags=[[m_mode_number[2], amp_fre_signal[7], amp_fre_signal[10]]],
#                                         save_repo=FileRepo(
#                                             resampled_shotsets_path)
#                                         )

# lock_mode 1052637(), 1055258, 1056287, 1057329, 1057336, 1057573, 1057832
###新增一个tag alarm_time?????????????
def disruptalarm(shot_no, lead_time, sample_rate = 200, dataset):
      shot_alarm = dataset.get_shot(shot_no)
      alarm_time = shot_alarm.labels["DownTime"] +lead_time
      undisrupt_number = int(sample_rate * alarm_time)
      signal_new = shot_alarm.get(m_mode_number[6])

      new_data = np.array([bool(0) for i in range(undisrupt_number)])
      new_data = np.append(new_data, np.array([bool(1) for i in range(len(signal_new.time)-undisrupt_number)]), axis=0)
      signal_new.data = new_data
      shot_alarm.add('\\signal_alram', signal_new)
      shot_alarm.save(FileRepo(resampled_shotsets_path))

# drop to_be_processed_path





