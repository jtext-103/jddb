import math
import pandas as pd
import os
from typing import Optional, List
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib


class Result:

    def __init__(self, csv_path: str):
        # check if file exist
        # if not create df,set threshold nan
        # if exists, read it populate self: call read
        self.tardy_alarm_threshold = np.nan
        self.lucky_guess_threshold = np.nan
        self.csv_path = csv_path
        self.__header = ['shot_no', 'predicted_disruption', 'predicted_disruption_time', 'true_disruption',
                         'true_disruption_time', 'warning_time', 'true_positive', 'false_positive',
                         'tardy_alarm_threshold', 'lucky_guess_threshold']
        self.result = None
        self.y_pred = []
        self.y_true = []
        self.shot_no = []
        self.ignore_thresholds = False
        if os.path.exists(csv_path):
            self.read()
        else:
            self.create_df()

    def create_df(self):
        # add header
        # set property
        df_result = pd.DataFrame(columns=self.__header)
        df_result.loc[1, ['tardy_alarm_threshold', 'lucky_guess_threshold']] = [self.tardy_alarm_threshold,
                                                                                self.lucky_guess_threshold]
        self.result = df_result
        self.save()

    def read(self):

        # check file format
        # read in         self.tardy_alarm_threshold  (make it property, in setter call calc_metrics)
        #                 self.lucky_guess_threshold
        # read in all shot
        # (call calc metric)
        self.result = pd.read_excel(self.csv_path)
        if self.result.columns is None:
            self.result = pd.DataFrame(columns=self.__header)
        elif len(set(self.result.columns) & set(self.__header)) != len(self.result.columns):
            raise ValueError("The file from csv_path:{} contains unknown information ".format(self.csv_path))

        self.tardy_alarm_threshold = self.result.loc[1, 'tardy_alarm_threshold']
        self.lucky_guess_threshold = self.result.loc[1, 'lucky_guess_threshold']

    def save(self):
        self.result.loc[1, ['lucky_guess_threshold']] = self.lucky_guess_threshold
        self.result.loc[1, ['tardy_alarm_threshold']] = self.tardy_alarm_threshold
        self.result.to_excel(self.csv_path, index=False)

    def get_all_shots(self, include_no_truth=True):
        # get all shot_no
        # if include_no_truth=True ,return shot_no without no_true
        # return shot_no
        shot_no = self.result.shot_no.tolist()
        if include_no_truth is False:
            get_shots = []
            for i in range(len(shot_no)):
                true_disruption_time = \
                    self.result.loc[self.result.shot_no == shot_no[i], 'true_disruption_time'].tolist()[0]
                if true_disruption_time is not None:
                    get_shots.append(shot_no[i])
            shot_no = get_shots
        return shot_no

    def check_unexisted(self, shot_no: Optional[List[int]] = None):
        # check whehter shot_no existed
        # if unexisted, raise error
        # if shot_no is None ,call get all shots()
        # return shot_no

        err_list = []
        for i in range(len(shot_no)):
            if shot_no[i] not in self.result.shot_no.tolist():
                err_list.append(shot_no[i])
            if len(err_list) > 0:
                raise ValueError("THE data of number or numbers:{} do not exist".format(err_list))
        if shot_no is None:
            shot_no = self.get_all_shots(include_no_truth=True)
        return shot_no

    def check_repeated(self, shot_no: Optional[List[int]]):
        # check whehter shot_no repeated
        # if repeated, raise error
        err_list = []
        for i in range(len(shot_no)):
            if shot_no[i] in self.result.shot_no.tolist():
                err_list.append(shot_no[i])
        if len(err_list) > 0:
            raise ValueError("data of shot_no:{} has already existed".format(err_list))

    def add(self, shot_no: List[int], predicted_disruption: List[bool], predicted_disruption_time: List[float]):
        # check lenth
        # check repeated shoot,call check_repeated()
        # use returned shot_no to add

        "增加指定数据，输入表中已有number，会报错"
        if not (len(shot_no) == len(predicted_disruption) == len(predicted_disruption_time)):
            raise ValueError('The inputs do not share the same length.')

        self.check_repeated(shot_no)
        for i in range(len(shot_no)):
            row_index = len(self.result) + 1  # 当前excel内容有几行
            if self.result.loc[self.result.shot_no == shot_no[i], 'predited_disruption'].tolist()[0] == 0:
                predicted_disruption_time[i] = -1
            self.result.loc[row_index, ['shot_no', 'predited_disruption', 'predicted_disruption_time']] = \
                [shot_no[i], predicted_disruption[i], predicted_disruption_time[i]]

    def remove(self, df_result, shot_no: List[int]):
        shot_no = self.check_unexisted(shot_no)
        for i in range(len(shot_no)):
            self.result = self.result.drop(df_result[self.result.shot_no == shot_no[i]].index)

    def get_all_truth(self, shot_no: Optional[List[str]], true_disruption: List[bool],
                      true_disruption_time: List[float]):
        # check input shot_no whether exist
        # add true data
        shot_no = self.check_unexisted(shot_no)

        for i in range(len(shot_no)):
            if self.result.loc[self.result.shot_no == shot_no[i], 'true_disruption'].tolist()[0] == 0:
                true_disruption_time[i] = -1
            self.result.loc[
                self.result[self.result.shot_no == shot_no[i]].index[0], ['true_disruption', 'true_disruption_time']] = \
                [true_disruption[i], true_disruption_time[i]]

    def get_y(self):

        # in put threshold before call judge_y_pred
        # check threshold
        # compute warning time
        # compute y_pred
        if math.isnan(self.tardy_alarm_threshold) or math.isnan(self.lucky_guess_threshold):
            raise ValueError(
                "tardy_alarm_threshold is :{} , lucky_guess_threshold is :{}, fulfill ".format(
                    self.tardy_alarm_threshold, self.lucky_guess_threshold))
        self.shot_no = self.get_all_shots(include_no_truth=False)
        self.get_warning_time()
        self.get_y_pred()
        self.get_y_true()

    def get_warning_time(self):
        shot_no = self.shot_no
        for i in range(len(shot_no)):
            predicted_disruption_time = \
                self.result.loc[self.result.shot_no == shot_no[i], 'predicted_disruption_time'].tolist()[0]
            true_disruption_time = \
                self.result.loc[self.result.shot_no == shot_no[i], 'true_disruption_time'].tolist()[0]
            self.result.loc[self.result.shot_no == shot_no[
                i], 'warning_time'] = true_disruption_time - predicted_disruption_time

    def get_y_pred(self):
        y_pred = []
        shot_no = self.shot_no

        for i in range(len(shot_no)):
            warning_time = self.result.loc[self.result.shot_no == shot_no[i], 'warning_time'].tolist()[0]
            predicted_disruption = self.result.loc[self.result.shot_no == shot_no[i], 'predicted_disruption'].tolist()[
                0]
            true_disruption = self.result.loc[self.result.shot_no == shot_no[i], 'true_disruption'].tolist()[0]
            if predicted_disruption == 1 and true_disruption == 1:
                if self.tardy_alarm_threshold < warning_time < self.lucky_guess_threshold:
                    y_pred.append(1)
                else:
                    if self.ignore_thresholds is False:
                        self.result.loc[self.result.shot_no == shot_no[i], 'warning_time'] = -1
            else:
                y_pred.append(0)
                self.result.loc[self.result.shot_no == shot_no[i], 'warning_time'] = -1
        self.y_pred = y_pred

    def get_y_true(self):
        # whether self.shot_no = shot_no
        # get y_pred, shot_no
        # get y_true
        shot_no = self.shot_no
        for i in range(len(shot_no)):
            self.y_true.append(
                self.result.loc[self.result.shot_no == shot_no[i], 'true_disruption'].tolist()[0])

    def calc_metrics(self):
        # in put threshold before call calc_metrics
        # whether self.shot_no = shot_no
        # get y_pred, shot_no
        # compute warning_time, true_positive, false_positive
        if len(set(self.get_all_shots(include_no_truth=False)) & set(self.shot_no)) != len(self.shot_no):
            self.get_y()
        shot_no = self.shot_no
        y_pred = self.y_pred

        for i in range(len(shot_no)):
            true_disruption = self.result.loc[self.result.shot_no == shot_no[i], 'true_disruption'].tolist()[0]
            if true_disruption == 1 and y_pred[i] == 1:
                self.result.loc[self.result.shot_no == shot_no[i], 'true_positive'] = 1
                self.result.loc[self.result.shot_no == shot_no[i], 'false_positive'] = 0

            elif true_disruption == 0 and y_pred[i] == 1:
                self.result.loc[self.result.shot_no == shot_no[i], 'true_positive'] = 0
                self.result.loc[self.result.shot_no == shot_no[i], 'false_positive'] = 1

            else:
                self.result.loc[self.result.shot_no == shot_no[i], 'true_positive'] = 0
                self.result.loc[self.result.shot_no == shot_no[i], 'false_positive'] = 0

    def confusion_matrix(self):
        # whether self.shot_no = shot_no
        # get y_pred, shot_no
        # compute confusion_matrix

        if len(set(self.get_all_shots(include_no_truth=False)) & set(self.shot_no)) != len(self.shot_no):
            self.get_y()
        [[tp, fn], [fp, tn]] = confusion_matrix(self.y_true, self.y_pred, labels=[1, 0])
        return tp, fn, fp, tn

    def ture_positive_rate(self):
        # get tp, fn, fp, tn
        # compute tpr, fpr
        tp, fn, fp, tn = self.confusion_matrix()
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        return tpr, fpr

    def get_accuracy(self):
        if len(set(self.get_all_shots(include_no_truth=False)) & set(self.shot_no)) != len(self.shot_no):
            self.get_y()
        accuracy = accuracy_score(y_true=self.y_true, y_pred=self.y_pred, normalize=True,
                                  sample_weight=None)
        return accuracy

    def get_precision(self):
        if len(set(self.get_all_shots(include_no_truth=False)) & set(self.shot_no)) != len(self.shot_no):
            self.get_y()

        precision = precision_score(y_true=self.y_true, y_pred=self.y_pred, average='macro')
        return precision

    def get_recall(self):
        if len(set(self.get_all_shots(include_no_truth=False)) & set(self.shot_no)) != len(self.shot_no):
            self.get_y()

        recall = recall_score(y_true=self.y_true, y_pred=self.y_pred, average='macro')
        return recall

    def warning_time_histogram(self, file_path: str, time_bins: List[float]):
        # file_path is the path to save .png
        # time_bins is a time endpoint list
        matplotlib.use('TkAgg')
        plt.style.use("ggplot")
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.weight'] = 'bold'
        warning_time_list = []
        shot_no = self.get_all_shots(include_no_truth=False)
        if len(set(self.get_all_shots(include_no_truth=False)) & set(self.shot_no)) != len(self.shot_no):
            self.get_y()
        for i in range(len(shot_no)):
            if self.result.loc[self.result.shot_no == shot_no[i], 'warning_time'].tolist()[0] != -1:
                warning_time_list.append(self.result.loc[self.result.shot_no == shot_no[i], 'warning_time'].tolist()[0])
        warning_time_list = np.array(warning_time_list)  # s#预测时间

        time_segments = pd.cut(warning_time_list, time_bins, right=False)
        print(time_segments)
        counts = pd.value_counts(time_segments, sort=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(
            x=counts.index.astype(str),
            height=counts,
            width=0.2,
            align="center",
            color="cornflowerblue",
            edgecolor="darkblue",
            linewidth=2.0
        )
        ax.set_title("warning_time_histogram", fontsize=15)
        plt.savefig(os.path.join(file_path, 'histogram_warning_time.png'), dpi=300)

    def accumulate_warning_time_plot(self, output_dir: str, true_dis_num):  #

        matplotlib.use('TkAgg')
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.weight'] = 'bold'
        warning_time_list = []
        shot_no = self.get_all_shots(include_no_truth=False)
        if len(set(self.get_all_shots(include_no_truth=False)) & set(self.shot_no)) != len(self.shot_no):
            self.get_y()
        for i in range(len(shot_no)):
            if self.result.loc[self.result.shot_no == shot_no[i], 'warning_time'].tolist()[0] != -1:
                warning_time_list.append(self.result.loc[self.result.shot_no == shot_no[i], 'warning_time'].tolist()[0])
        warning_time = np.array(warning_time_list)  # ms->s#预测时间
        warning_time.sort()  #
        accu_frac = list()
        for i in range(len(warning_time)):
            accu_frac.append((i + 1) / true_dis_num * 100)
        axis_width = 2
        major_tick_length = 12
        minor_tick_length = 6
        fig, ax = plt.subplots()
        fig.set_figheight(7.5)
        fig.set_figwidth(10)
        ax.set_xscale('log')
        ax.set_xlabel('Warning Time (s)', fontweight='bold')
        ax.set_ylabel('Accumulated Disruptions Predicted (%)', fontweight='bold')

        ax.plot(warning_time[::-1], accu_frac, color="cornflowerblue", linewidth=3.5)
        ax.spines['bottom'].set_linewidth(axis_width)
        ax.spines['top'].set_linewidth(axis_width)
        ax.spines['left'].set_linewidth(axis_width)
        ax.spines['right'].set_linewidth(axis_width)
        y_major_locator = MultipleLocator(10)
        ax.yaxis.set_major_locator(y_major_locator)
        y_minor_locator = MultipleLocator(5)
        ax.yaxis.set_minor_locator(y_minor_locator)
        ax.tick_params(which='both', direction='in', width=axis_width)
        ax.tick_params(which='major', length=major_tick_length)
        ax.tick_params(which='minor', length=minor_tick_length)

        plt.savefig(os.path.join(output_dir, 'accumulate_warning_time.png'), dpi=300)

