# %% import
import json
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, ClipProcessor, TrimProcessor
from matplotlib import pyplot as plt
from basic_processor import *
import os

# %% set up some const

# tags that need to be normalized, the mean and std. are saved in config files
normalize_tags = [
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

# tags to be kept after all the processing
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

# slicing parameters
window_length = 2500
resampled_rate = 1000
overlap = 0.9
# clip the signal at
clip_start_time = 0.05

# file repo paths
source_shots_path = "..//FileRepo//TestShots//$shot_2$XX//$shot_1$X//"
processed_shots_path = "..//FileRepo//ProcessedShots//$shot_2$XX//$shot_1$X//"
image_path = "..//FileRepo//_temp_image//"

# %% define some helper functions

# read a dict from json, in this example the config file stores the
# mean and std for some signals


def read_config(file_name: str):
    """"read config files"""
    with open(file_name, 'r', encoding='UTF-8') as f:
        config = json.load(f)
    return config


def find_tags(prefix, all_tags):
    """
        find tags that start with the prefix
    param:
        prefix: The first few strings of the tags users need to look for
        all_tags: a list of all the tags that needed to be filtered
    :return: matching tags as a list[sting]
    """
    return list(filter(lambda tag: tag.encode("utf-8").decode("utf-8", "ignore")[0:len(prefix)] == prefix, all_tags))


# %% get source and processed filerepo ready
if __name__ == '__main__':
    # read mean and std for normalization
    config_mean = read_config("config_mean.json")
    config_std = read_config("config_std.json")

    # get file repo
    source_file_repo = FileRepo(source_shots_path)
    processed_file_repo = FileRepo(processed_shots_path)
    # create a shot set with a file
    source_shotset = ShotSet(source_file_repo)
    # get all shots and tags from the file repo
    shot_list = source_shotset.shot_list
    all_tags = list(source_shotset.get_shot(shot_list[0]).tags)

    # %% [markdown]
    # below are steps to extract features

    # %%
    # 1.slicing
    # generate a new signal with is moving window slices of a source signal
    # it is for FFT processors
    processed_shotset = source_shotset.process(
        processor=SliceProcessor(window_length=window_length, overlap=overlap),
        input_tags=["\\MA_POL2_P01"],
        output_tags=["\\sliced_MA"], save_repo=processed_file_repo)

    # %%
    # 2. fft MA mir array signal
    processed_shotset = processed_shotset.process(
        processor=FFTProcessor(),
        input_tags=["\\sliced_MA"],
        output_tags=[["\\fft_amp", "\\fft_fre"]],
        save_repo=processed_file_repo)

    # %%
    # 3.mean
    # calculate average of an array diagnostics output to a new signal

    # average density
    den = find_tags("\\polar", all_tags)
    processed_shotset = processed_shotset.process(processor=Mean(),
                                                  input_tags=[den],
                                                  output_tags=[
                                                      "\\polaris_den_mean"],
                                                  save_repo=processed_file_repo)
    # average soft s-ray
    sxr = find_tags("\\sxr_c", all_tags)

    processed_shotset = processed_shotset.process(processor=Mean(),
                                                  input_tags=[sxr],
                                                  output_tags=["\\sxr_c_mean"],
                                                  save_repo=processed_file_repo)
    # %%
    # 5. resample all_tags
    # down sample to 1kHz
    all_tags = (list(processed_shotset.get_shot(shot_list[0]).tags))
    # sliced signalneed not to be resampled
    all_tags.remove("\\sliced_MA")
    processed_shotset = processed_shotset.process(processor=ResamplingProcessor(resampled_rate),
                                                  input_tags=all_tags,
                                                  output_tags=all_tags,
                                                  save_repo=processed_file_repo)
    # %%
    # 6. normalization all raw signals
    for tag in normalize_tags:
        mean = config_mean[tag]
        std = config_std[tag]

        processed_shotset = processed_shotset.process(
            processor=NormalizationProcessor(mean=float(mean), std=float(std)),
            input_tags=[tag],
            output_tags=[tag],
            save_repo=processed_file_repo)

    # %%
    # 7.drop useless raw signals
    exsads = find_tags("\\exsad", all_tags)
    processed_shotset = processed_shotset.remove_signal(tags=keep_tags + exsads, keep=True,
                                                        save_repo=processed_file_repo)
    # %%
    # 8.clip ,remove signal out side of the time of interests
    # get the new set of tags, after the processing the tags have changed
    all_tags = list(processed_shotset.get_shot(shot_list[0]).tags)
    processed_shotset = processed_shotset.process(
        processor=ClipProcessor(
            start_time=clip_start_time, end_time_label="DownTime"),
        input_tags=all_tags,
        output_tags=all_tags,
        save_repo=processed_file_repo)

    # %%
    # 9. add disruption labels for each time point as a signal called alarm_tag
    processed_shotset = processed_shotset.process(
        processor=AlarmTag(
            lead_time=0.01, disruption_label="IsDisrupt", downtime_label="DownTime"),
        input_tags=["\\ip"], output_tags=["\\alram_tag"],
        save_repo=processed_file_repo)

    # %%
    # 10. trim all signal
    all_tags = list(processed_shotset.get_shot(shot_list[0]).tags)
    processed_shotset = processed_shotset.process(TrimProcessor(),
                                                  input_tags=[all_tags],
                                                  output_tags=[all_tags],
                                                  save_repo=processed_file_repo)

    # %%
    # plot the result
    # p1 ip
    # p2 fft
    shot_plt = processed_shotset.get_shot(shot_list[5])
    signal_ip = shot_plt.get_signal("\\ip")  # p1
    signal_MA_amp = shot_plt.get_signal("\\fft_amp")  # p2
    signal_MA_fre = shot_plt.get_signal("\\fft_fre")  # p2
    end_time = signal_MA_amp.time[-1]

    # save the plot
    dir_name = os.path.join(image_path, str(shot_list[0]) + '.jpg')
    fig = plt.figure(1, figsize=(23, 13))

    plt.subplots_adjust(wspace=0.4, hspace=0.2)

    # "ip"
    plt.subplot(2, 1, 1)  # p1
    plt.plot(signal_ip.time, signal_ip.data)
    plt.xlim([0, end_time])
    plt.ylabel("ip")

    # "amp_fre"
    ax1 = plt.subplot(2, 1, 2)  # p2
    ax2 = ax1.twinx()
    ax1.plot(signal_MA_amp.time, signal_MA_amp.data, 'b-')  # amp
    ax2.plot(signal_MA_amp.time, signal_MA_fre.data, 'r-')  # fre
    plt.xlim([0, end_time])
    plt.xlabel("time")
    ax1.set_ylabel("Max_Amp", color='b')
    ax2.set_ylabel("fre", color='r')
    plt.grid(axis="x", linestyle='-.', which='major')
    plt.suptitle(str(shot_list[0]) + "_5ms", x=0.5, y=0.98)
    plt.show()

# %%
