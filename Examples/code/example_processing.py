from jddb.processor import ShotSet, Shot, Signal, BaseProcessor
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor
from jddb.file_repo import FileRepo
import matplotlib
from matplotlib import pyplot as plt
from example_processor import *
import numpy as np
import os


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
clip_second = 0.05
overlap = 0.9

raw_shotsets_path = "G:\\datapractice\\example_lry\\Example_shot_files"
processed_path = "G:\\datapractice\\example_lry\\example_result_data\\processed_shotsets"
processing_shotsets_path = "G:\\datapractice\\example_lry\\example_result_data\\processing_shotsets"
picture_path = "G:\\datapractice\\example_lry\\example_result_data\\fre_picture"


def get_parameter(tag, dataset):
    """
        the tag are fed into the function get_parameter, which concatenated by array of a certain kind of diagnosis
        inside each shot.
        concatenate the input tag shot by shot, getting the kind of diagnosis  of the whole dataset.
        return the calculate the standard deviation and mean. called by normalization
    """
    data_concat = np.empty(shape=0, dtype=np.float32)
    for i in range(len(shotlist)):
        shot_concat = dataset.get_shot(shotlist[i])
        concat_signal = shot_concat.get(tag)
        data_concat = np.concatenate((data_concat, concat_signal.data))
    mean = np.mean(data_concat, dtype=np.float32)
    std = np.std(data_concat, dtype=np.float32)
    return mean, std


def normalization(n_mode_dataset, to_concat_list, concated_list, normal_list):
    """
        Normalize the signal
    :param n_mode_dataset:

    :param to_concat_list: the tags needed to concatenate before calculating the standard deviation
    :param concated_list: output new concatenated tags in intermediate steps of normalization

    :param normal_list: the tags to be normalized, call NormalizationProcessor to process
    """
    for index in range(len(to_concat_list)):
        out_put = [concated_list[index]]
        n_mode_dataset = n_mode_dataset.process(processor=Concatenate(),
                                                input_tags=[to_concat_list[index]],
                                                output_tags=out_put,
                                                save_repo=FileRepo(
                                                    processing_shotsets_path))
        mean, std = get_parameter(concated_list[index], n_mode_dataset)
        if isinstance(normal_list[index], str):
            normal_list[index] = eval(str(list(normal_list[index])).replace(',', '').replace(' ', ''))
        n_mode_dataset = n_mode_dataset.process(processor=NormalizationProcessor(mean, std),
                                                input_tags=normal_list[index],
                                                output_tags=normal_list[index],
                                                save_repo=FileRepo(
                                                    processing_shotsets_path))

def find_tags(string):
    """
        Given the string lookup, the output contain tags with the same content
    :param string: "string"
    :return: tags with the same content
    """
    return list(filter(lambda tag: tag.encode("utf-8").decode("utf-8", "ignore")[0:len(string)] == string, all_tags))


if __name__ == '__main__':
    #%%load data
    base_path = os.path.join(raw_shotsets_path, "$shot_2$XX", "$shot_1$X")
    file_repo = FileRepo(base_path)
    n_mode_dataset = ShotSet(file_repo)
    processing_shotsets_path = os.path.join(processing_shotsets_path, "$shot_2$00", "$shot_1$0")
    processed_path = os.path.join(processed_path, "$shot_2$00", "$shot_1$0")
    # n_mode_dataset = ShotSet(FileRepo(processing_shotsets_path))
    shotlist = n_mode_dataset.shot_list
    all_tags = list(n_mode_dataset.get_shot(shotlist[0]).tags)

    #%%
    # # 1.clip
    n_mode_dataset = n_mode_dataset.process(processor=TimeClipProcessor(start_time=0),
                                            input_tags=all_tags,
                                            output_tags=all_tags,
                                            save_repo=FileRepo(
                                                processing_shotsets_path))
    # %%

    # %%
    # # 2.slice MA
    n_mode_dataset = n_mode_dataset.process(processor=SliceProcessor(window_length=window_length, overlap=overlap),
                                            input_tags=["\\MA_POL2_P01"],
                                            output_tags=["\\fft_amp"],
                                            save_repo=FileRepo(
                                                    processing_shotsets_path))
    # %%

    # %%
    # # 3. fft MA
    n_mode_dataset = n_mode_dataset.process(processor=FFTProcessor(),
                                            input_tags=[["\\fft_amp", "\\MA_POL2_P01"]],
                                            output_tags=[["\\fft_amp", "\\fft_fre"]],
                                            save_repo=FileRepo(
                                                processing_shotsets_path))
    # %%
    #
    # %%
    # # 4.mean
    # # calc mean of "'\\polaris_den_v...'"
    #
    den = find_tags("\\polar")
    n_mode_dataset = n_mode_dataset.process(processor=Mean(),
                                            input_tags=[den],
                                            output_tags=["\\polaris_den_mean"],
                                            save_repo=FileRepo(
                                                processing_shotsets_path))
    #
    # %%

    # %%
    # # calc mean of "'\\sxr_cc...'"
    sxr = find_tags("\\sxr_c")
    n_mode_dataset = n_mode_dataset.process(processor=Mean(),
                                            input_tags=[sxr],
                                            output_tags=["\\sxr_c_mean"],
                                            save_repo=FileRepo(
                                                processing_shotsets_path))
    # %%

    # %%
    # # 5. resample all_tags
    all_tags = list(n_mode_dataset.get_shot(shotlist[0]).tags)
    n_mode_dataset = n_mode_dataset.process(processor=ResamplingProcessor(resampled_rate),
                                            input_tags=all_tags,
                                            output_tags=all_tags,
                                            save_repo=FileRepo(
                                                processing_shotsets_path))
    # %%

    # %%
    # #6. normalization all(Ivfp,Ihfp)
    # Each shot generates a concatenate_signal
    # Read the concatenate_signal of each cannon and concatenate
    exsad = find_tags("\\exsad")

    to_concat_list = [
        ["\\Ivfp", "\\Ihfp"],
        "\\Iohp",
        "\\bt",
        "\\dx",
        "\\dy",
        exsad,
        "\\ip",
        "\\vl",
        den,
        sxr,
        "\\fft_amp",
        "\\fft_fre"]
    concated_list = [
        "\\Ifp_contac",
        "\\Iohp",
        "\\bt",
        "\\dx",
        "\\dy",
        "\\exs_contac",
        "\\ip",
        "\\vl",
        "\\den_contac",
        "\\sxr_contac",
        "\\fft_amp",
        "\\fft_fre"
    ]
    normal_list = [
        ["\\Ivfp", "\\Ihfp"],
        "\\Iohp",
        "\\bt",
        "\\dx",
        "\\dy",
        exsad,
        "\\ip",
        "\\vl",
        "\\polaris_den_mean",
        "\\sxr_c_mean",
        "\\fft_amp",
        "\\fft_fre"]

    normalization(n_mode_dataset, to_concat_list, concated_list, normal_list)
    # %%

    # %%
    # # 7.drop siganl
    exsad = find_tags("\\exsad")
    n_mode_dataset = n_mode_dataset.remove(tags=keep_tags + exsad, keep=True,
                                           save_repo=FileRepo(processed_path))
    # %%

    # %%
    # # 8.clip
    all_tags = n_mode_dataset.get_shot(shotlist[0]).tags
    n_mode_dataset = n_mode_dataset.process(processor=TimeClipProcessor(start_time=0),
                                            input_tags=all_tags,
                                            output_tags=all_tags,
                                            save_repo=FileRepo(
                                                processed_path))
    # %%

    # %%
    # # 9.alarm_tag
    n_mode_dataset = n_mode_dataset.process(processor=AlarmTag(lead_time=0.01, fs=1000, start_time=0),
                                            input_tags=["\\ip"],
                                            output_tags=["\\alramtag"],
                                            save_repo=FileRepo(
                                                processed_path))
    #%%

    #%%picture
    #
    # p1 ip
    # p2 fft
    shot_plt = n_mode_dataset.get_shot(shotlist[0])
    signal_ip = shot_plt.get("\\ip")  # p1
    signal_MA_amp = shot_plt.get("\\fft_amp")  # p2
    signal_MA_fre = shot_plt.get("\\fft_fre")  # p2
    end_time = signal_MA_amp.time[-1]

    # Draw a graph and save the specified path
    # Image saving path
    dir_name = os.path.join(picture_path, str(shotlist[0]) + '.jpg')
    fig = plt.figure(1, figsize=(23, 13))

    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    # "ip"
    plt.subplot(2, 1, 1)  # p1
    plt.plot(signal_ip.time, signal_ip.data)
    plt.xlim([0, end_time])
    plt.ylabel("ip")

    #"amp_fre"
    ax1 = plt.subplot(2, 1, 2)  # p2
    ax2 = ax1.twinx()
    ax1.plot(signal_MA_amp.time, signal_MA_amp.data, 'b-')  # amp
    ax2.plot(signal_MA_amp.time, signal_MA_fre.data, 'r-')  # fre
    plt.xlim([0, end_time])
    plt.xlabel("time")
    ax1.set_ylabel("Max_Amp", color='b')
    ax2.set_ylabel("fre", color='r')
    plt.grid(axis="x", linestyle='-.', which='major')
    plt.suptitle(str(shotlist[0]) + "_5ms", x=0.5, y=0.98)
    plt.savefig(dir_name)
    plt.show()




