import json

from jddb.processor import ShotSet, Shot, Signal, BaseProcessor
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor
from jddb.file_repo import FileRepo
import matplotlib
from matplotlib import pyplot as plt
from example_processor import *
import numpy as np
import os

get_mean_tags=[
  "\\Ivfp",
  "\\Ihfp",
  "\\Iohp",
  "\\bt",
  "\\dx",
  "\\dy",
  "\\exsad7",
  "\\exsad10",
  "\\exsad4",
  "\\exsad1",
  "\\ip",
  "\\vl",
  "\\polaris_den_mean",
  "\\sxr_c_mean",
  "\\fft_amp",
  "\\fft_fre"
]

keep_tags = [
    "\\Ivfp",
    "\\Ihfp",
    "\\Iohp",
    "\\bt",
    "\\dx",
    "\\dy",
    "\\ip",
    "\\vl",
    "\\polaris_den_mean",
    "\\sxr_c_mean",
    "\\fft_amp",
    "\\fft_fre",
]

window_length = 2500
resampled_rate = 1000
clip_start_time = 0.05
overlap = 0.9

source_shots_path = "G:\\datapractice\\example_lry\\Example_shot_files"
processed_shots_path = "G:\\datapractice\\example_lry\\example_result_data\\processed_shotsets"
picture_path = "G:\\datapractice\\example_lry\\example_result_data\\fre_picture"
source_shots_path = os.path.join(source_shots_path, "$shot_2$XX", "$shot_1$X")
processed_shots_path = os.path.join(processed_shots_path, "$shot_2$00", "$shot_1$0")

def read_config(file_name:str):
    """"读取配置"""
    with open(file_name, 'r', encoding='UTF-8') as f:
        config = json.load(f)
    return config

def find_tags(string):
    """
        Given the string lookup, the output contain tags with the same content
    :param string: "string"
    :return: tags with the same content
    """
    return list(filter(lambda tag: tag.encode("utf-8").decode("utf-8", "ignore")[0:len(string)] == string, all_tags))


if __name__ == '__main__':
    # %%load data
    config_mean = read_config("config_mean.json")
    config_std = read_config("config_std.json")
    source_file_repo = FileRepo(source_shots_path)
    processed_file_repo = FileRepo(processed_shots_path)
    # source_shotset = ShotSet(source_file_repo)
    # shotlist = source_shotset.shot_list
    # all_tags = list(source_shotset.get_shot(shotlist[0]).tags)
    processed_shotset = ShotSet(processed_file_repo)
    shotlist = processed_shotset.shot_list
    # all_tags = list(processed_shotset.get_shot(shotlist[0]).tags)

    # %%
    # # 1.clip
    # processed_shotset = source_shotset.process(processor=TimeTrimProcessor(start_time=clip_start_time, end_time_tag=""),
    #                                            input_tags=all_tags,
    #                                            output_tags=all_tags, save_repo=processed_file_repo)
    # %%

    # # %%
    # # # 2.slice MA
    # processed_shotset = source_shotset.process(
    #     processor=SliceProcessor(window_length=window_length, overlap=overlap), input_tags=["\\MA_POL2_P01"],
    #     output_tags=["\\sliced_MA"], save_repo=processed_file_repo)
    # # %%
    #
    # # %%
    # # # 3. fft MA
    # processed_shotset = processed_shotset.process(processor=FFTProcessor(),
    #                                               input_tags=[["\\sliced_MA", "\\MA_POL2_P01"]],
    #                                               output_tags=[["\\fft_amp", "\\fft_fre"]],
    #                                               save_repo=processed_file_repo)
    # # %%
    # #
    # # %%
    # # # 4.mean
    # # # calc mean of "'\\polaris_den_v...'"
    # #
    # den = find_tags("\\polar")
    # processed_shotset = processed_shotset.process(processor=Mean(), input_tags=[den],
    #                                               output_tags=["\\polaris_den_mean"],
    #                                               save_repo=processed_file_repo)
    # # #
    # # # %%
    #
    # # %%
    # # # calc mean of "'\\sxr_cc...'"
    # sxr = find_tags("\\sxr_c")
    # processed_shotset = processed_shotset.process(processor=Mean(), input_tags=[sxr],
    #                                               output_tags=["\\sxr_c_mean"],
    #                                               save_repo=
    #                                               processed_file_repo)
    # # %%
    #
    # # %%
    # # # 5. resample all_tags
    # all_tags = (list(processed_shotset.get_shot(shotlist[0]).tags))
    # all_tags.remove("\\sliced_MA")
    # processed_shotset = processed_shotset.process(processor=ResamplingProcessor(resampled_rate),
    #                                               input_tags=all_tags,
    #                                               output_tags=all_tags,
    #                                               save_repo=processed_file_repo)
    # # %%

    # # %%
    # # #6. normalization all(Ivfp,Ihfp)
    # # Each shot generates a concatenate_signal
    # # Read the concatenate_signal of each cannon and concatenate



    # for index in range(len(get_mean_tags)):
    #     mean = config_mean[get_mean_tags[index]]
    #     std = config_std[get_mean_tags[index]]
    #
    #     processed_shotset = processed_shotset.process(processor=NormalizationProcessor(mean=float(mean), std=float(std)),
    #                                                   input_tags=[get_mean_tags[index]],
    #                                                   output_tags=[get_mean_tags[index]],
    #                                                   save_repo=processed_file_repo)

    # %%

    # %%
    # # 7.drop siganl
    # exsad = find_tags("\\exsad")
    # processed_shotset = processed_shotset.remove(tags=keep_tags + exsad, keep=True,
    #                                              save_repo=processed_file_repo)
    # %%

    # %%
    # # # 8.clip
    # all_tags = processed_shotset.get_shot(shotlist[0]).tags
    # processed_shotset = processed_shotset.process(
    #     processor=TimeTrimProcessor(start_time=clip_start_time, end_time_tag="DownTime"),
    #     input_tags=all_tags,
    #     output_tags=all_tags, save_repo=processed_file_repo)
    # # %%

    # %%
    # # 9.alarm_tag
    processed_shotset = processed_shotset.process(
        processor=AlarmTag(lead_time=0.01, disruption_label="IsDisrupt", downtime_label="DownTime"),
        input_tags=["\\ip"], output_tags=["\\alramtag"],
        save_repo=processed_file_repo)
    # # %%
    #
    # # %%picture
    # #
    # # p1 ip
    # # p2 fft
    # shot_plt = processed_shotset.get_shot(shotlist[0])
    # signal_ip = shot_plt.get("\\ip")  # p1
    # signal_MA_amp = shot_plt.get("\\fft_amp")  # p2
    # signal_MA_fre = shot_plt.get("\\fft_fre")  # p2
    # end_time = signal_MA_amp.time[-1]
    #
    # # Draw a graph and save the specified path
    # # Image saving path
    # dir_name = os.path.join(picture_path, str(shotlist[0]) + '.jpg')
    # fig = plt.figure(1, figsize=(23, 13))
    #
    # plt.subplots_adjust(wspace=0.4, hspace=0.2)
    # # "ip"
    # plt.subplot(2, 1, 1)  # p1
    # plt.plot(signal_ip.time, signal_ip.data)
    # plt.xlim([0, end_time])
    # plt.ylabel("ip")
    #
    # # "amp_fre"
    # ax1 = plt.subplot(2, 1, 2)  # p2
    # ax2 = ax1.twinx()
    # ax1.plot(signal_MA_amp.time, signal_MA_amp.data, 'b-')  # amp
    # ax2.plot(signal_MA_amp.time, signal_MA_fre.data, 'r-')  # fre
    # plt.xlim([0, end_time])
    # plt.xlabel("time")
    # ax1.set_ylabel("Max_Amp", color='b')
    # ax2.set_ylabel("fre", color='r')
    # plt.grid(axis="x", linestyle='-.', which='major')
    # plt.suptitle(str(shotlist[0]) + "_5ms", x=0.5, y=0.98)
    # plt.show()
