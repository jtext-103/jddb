import math
import os
from typing import List
from typing import Optional
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from ..meta_db import MetaDB
from ..file_repo import FileRepo


class Result:
    """
        define header
    """
    IS_DISRUPT = "IsDisrupt"
    DOWN_TIME = "DownTime"
    SHOT_NO_H = 'shot_no'
    PREDICTED_DISRUPTION_H = 'predicted_disruption'
    PREDICTED_DISRUPTION_TIME_H = 'predicted_disruption_time'
    ACTUAL_DISRUPTION_H = 'actual_disruption'
    ACTUAL_DISRUPTION_TIME_H = 'actual_disruption_time'
    WARNING_TIME_H = 'warning_time'
    TRUE_POSITIVE_H = 'true_positive'
    FALSE_POSITIVE_H = 'false_positive'
    TARDY_ALARM_THRESHOLD_H = 'tardy_alarm_threshold'
    LUCKY_GUESS_THRESHOLD_H = 'lucky_guess_threshold'

    def __init__(self, csv_path: str):
        """
            check if file exist
            if not create df,set threshold nan
            if exists, read it populate self: call read

        Args:
            csv_path: a path to read or create a csv file
        """

        self.tardy_alarm_threshold = np.nan
        self.lucky_guess_threshold = np.nan
        self.csv_path = csv_path
        self.__header = [self.SHOT_NO_H, self.PREDICTED_DISRUPTION_H, self.PREDICTED_DISRUPTION_TIME_H, self.ACTUAL_DISRUPTION_H,
                         self.ACTUAL_DISRUPTION_TIME_H, self.WARNING_TIME_H, self.TRUE_POSITIVE_H, self.FALSE_POSITIVE_H,
                         self.TARDY_ALARM_THRESHOLD_H, self.LUCKY_GUESS_THRESHOLD_H]
        self.result = None
        self.y_pred = []
        self.y_true = []
        self.shots = []
        self.ignore_thresholds = False
        if os.path.exists(csv_path):
            self.read()
        else:
            self.result = pd.DataFrame(columns=self.__header)

    def read(self):
        """
            check file format
            read in        self.tardy_alarm_threshold
                       self.lucky_guess_threshold
            read in all shot
        """

        self.result = pd.read_csv(self.csv_path)
        if self.result.columns is None:
            self.result = pd.DataFrame(columns=self.__header)
        elif len(set(self.result.columns) & set(self.__header)) != len(self.result.columns):
            raise ValueError("The file from csv_path:{} contains unknown information ".format(self.csv_path))
        self.tardy_alarm_threshold = self.result.loc[0, self.TARDY_ALARM_THRESHOLD_H]
        self.lucky_guess_threshold = self.result.loc[0, self.LUCKY_GUESS_THRESHOLD_H]

    def save(self):
        """
            after all result.save(),  the file will be saved in disk, else not
        """
        self.result.loc[0, [self.TARDY_ALARM_THRESHOLD_H]] = self.lucky_guess_threshold
        self.result.loc[0, [self.LUCKY_GUESS_THRESHOLD_H]] = self.tardy_alarm_threshold
        self.result.to_csv(self.csv_path, index=False)

    def get_all_shots(self, include_all=True):
        """
            get all shot_list return shot_list
        Args:
             include_all: if include_all=False, return shot_list without shots do
        not exist actual disruption information in this csv file.
        Returns:
            shot_list: a list of shot number

        """
        shot_list = self.result[self.SHOT_NO_H].tolist()
        if include_all is False:
            tmp_shot_list = []
            for shot_no in shot_list:
                true_disruption_time = self.result.loc[self.result[self.SHOT_NO_H] == shot_no, self.ACTUAL_DISRUPTION_TIME_H].tolist()[0]
                if true_disruption_time is not None:
                    tmp_shot_list.append(shot_no)
            shot_list = tmp_shot_list
        return shot_list

    def check_unexisted(self, shot_list: Optional[List[int]] = None):

        """
            check whehter shot_list existed
            if unexisted, raise error
            if shot_list is None ,call get all shots()
        """
        err_list = []
        for i in range(len(shot_list)):
            if shot_list[i] not in self.result[self.SHOT_NO_H].tolist():
                err_list.append(shot_list[i])
            if len(err_list) > 0:
                raise ValueError("THE data of number or numbers:{} do not exist".format(err_list))
        if shot_list is None:
            shot_list = self.get_all_shots(include_all=True)
        return shot_list

    def check_repeated(self, shot_list: Optional[List[int]]):
        """
            check whehter shot_list repeated
            if repeated, raise error
        Args:
            shot_list: a list of shot number
        """

        err_list = []
        for i in range(len(shot_list)):
            if shot_list[i] in self.result[self.SHOT_NO_H].tolist():
                err_list.append(shot_list[i])
        if len(err_list) > 0:
            raise ValueError("data of shot_list:{} has already existed".format(err_list))

    def add(self, shot_list: List[int], predicted_disruption: List[int], predicted_disruption_time: List[float]):
        """
            check lenth
            check repeated shot, if duplicated overwrite
        Args:
            shot_list: a list of shot number
            predicted_disruption:   a list of value 0 or 1, is disruptive
            predicted_disruption_time: a list of predicted_disruption_time, unit :s
        """
        if not (len(shot_list) == len(predicted_disruption) == len(predicted_disruption_time)):
            raise ValueError('The inputs do not share the same length.')

        for i in range(len(shot_list)):
            shot = shot_list[i]
            if predicted_disruption[i] == 0:
                predicted_disruption_time[i] = -1
            # check if key already exists in the dataframe
            if shot_list[i] in self.result[self.SHOT_NO_H].values:
                # update the row with matching key
                self.result.loc[self.result[self.SHOT_NO_H] == shot, [self.SHOT_NO_H, self.PREDICTED_DISRUPTION_H,
                                                                      self.PREDICTED_DISRUPTION_TIME_H]] = \
                    [shot_list[i], predicted_disruption[i], predicted_disruption_time[i]]
            else:
                # insert new row
                new_row = {self.SHOT_NO_H: int(shot_list[i]), self.PREDICTED_DISRUPTION_H: predicted_disruption[i],
                           self.PREDICTED_DISRUPTION_TIME_H: predicted_disruption_time[i]}
                self.result = self.result.append(new_row, ignore_index=True)

    def get_all_truth_from_metadb(self, meta_db: MetaDB):
        """
            get all true_disruption and true_downtime of exsit shot number from meta_db, if duplicated overwrite
        """


        shots = self.get_all_shots()
        for shot in shots:
            true_disruption = meta_db.get_labels(shot)[self.IS_DISRUPT]

            if true_disruption == False:
                true_disruption = 0
                true_downtime = -1
            else:
                true_disruption = 1
                true_downtime = meta_db.get_labels(shot)[self.DOWN_TIME]
            self.result.loc[self.result[self.SHOT_NO_H] == shot, self.ACTUAL_DISRUPTION_H] = true_disruption
            self.result.loc[self.result[self.SHOT_NO_H] == shot, self.ACTUAL_DISRUPTION_TIME_H] = true_downtime

    def get_all_from_file_repo(self, file_repo: FileRepo):
        """
            get all true_disruption and true_downtime of exsit shot number from file_repo, if duplicated overwrite
        """

        shots = file_repo.get_all_shots()
        for shot in shots:
            true_disruption = file_repo.read_labels(shot)[self.IS_DISRUPT]

            if true_disruption == False:
                true_disruption = 0
                true_downtime = -1
            else:
                true_disruption = 1
                true_downtime = file_repo.read_labels(shot)[self.DOWN_TIME]
            self.result.loc[self.result[self.SHOT_NO_H] == shot, self.ACTUAL_DISRUPTION_H] = true_disruption
            self.result.loc[self.result[self.SHOT_NO_H] == shot, self.ACTUAL_DISRUPTION_TIME_H] = true_downtime


    def remove(self, shot_list: List[int]):
        """

                giving model_name to remove the specified row
        Args:
            shot_list: a list of shot number


        """

        shot_list = self.check_unexisted(shot_list)
        for i in range(len(shot_list)):
            self.result = self.result.drop(self.result[self.result[self.SHOT_NO_H] == shot_list[i]].index)


    def get_y(self):
        """
                this function is called by self.calc_metrics()
                check threshold: if (self.tardy_alarm_threshold or self.lucky_guess_threshold) is nan, raise error,
        then the user should in put threshold BEFORE call self.get_y()
                compute warning time: a list of (true_disruption_time - predicted_disruption_time), unit: s
                compute y_pred: a list of value 0 or 1, 1 is right
        """

        if math.isnan(self.tardy_alarm_threshold) or math.isnan(self.lucky_guess_threshold):
            raise ValueError(
                "tardy_alarm_threshold is :{} , lucky_guess_threshold is :{}, fulfill ".format(
                    self.tardy_alarm_threshold, self.lucky_guess_threshold))
        self.shots = self.get_all_shots(include_all=False)
        self.get_warning_time()

        self.get_y_pred()
        self.get_y_true()

    def get_warning_time(self):
        """
                this function is called by self.get_y()
                compute warning time: a list of (true_disruption_time - predicted_disruption_time),
            unit:s
                There is a special case while true_disruption_time == -1 and predicted_disruption == 1,
            warning_time = predicted_disruption_time
        """
        shot_list = self.shots
        for i in range(len(shot_list)):
            predicted_disruption_time = \
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.PREDICTED_DISRUPTION_TIME_H].tolist()[0]
            true_disruption_time = \
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.ACTUAL_DISRUPTION_TIME_H].tolist()[0]
            predicted_disruption = \
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.PREDICTED_DISRUPTION_H].tolist()[0]
            if true_disruption_time == -1 and predicted_disruption == 1:
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[
                    i], self.WARNING_TIME_H] = predicted_disruption_time
            else:
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[
                    i], self.WARNING_TIME_H] = true_disruption_time - predicted_disruption_time

    def get_y_pred(self):
        """
                this function should be called before self.get_warning_time() ,and is called by self.get_y()
                giving self.ignore_threshold compute a value of true or false, then call get_y_pred to
            compute y_pred, a list of value 0 or 1, 1 is right , and revise warning_time:
                if ignore_thresholds is True,
                    set y_pred =1 while predicted_disruption == 1 ,
                    set y_pred = 0 and warning_time = -1 while predicted_disruption == 0
                if ignore_thresholds is False,
                    set y_pred =1 while predicted_disruption == 1 ,
                    set y_pred = 0 and warning_time = -1 while predicted_disruption == 0
        """
        y_pred = []
        shot_list = self.shots

        for i in range(len(shot_list)):
            warning_time = self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H].tolist()[0]
            predicted_disruption = self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.PREDICTED_DISRUPTION_H].tolist()[
                0]
            if predicted_disruption == 1:
                if self.ignore_thresholds is True:
                    y_pred.append(1)
                else:
                    if self.tardy_alarm_threshold < warning_time < self.lucky_guess_threshold:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
                        self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H] = -1
            else:
                y_pred.append(0)
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H] = -1
        self.y_pred = y_pred

    def get_y_true(self):
        """
            whether self.shots = shot_list
            get y_true: a list of value 0 or 1, 1 is right

        Returns:

        """

        y_true = []
        shot_list = self.shots
        y_true = []
        for i in range(len(shot_list)):
            y_true.append(
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.ACTUAL_DISRUPTION_H].tolist()[0])
        self.y_true = y_true

    def calc_metrics(self):
        """
            this function should be called before setting self.tardy_alarm_threshold and self.lucky_guess_threshold ,
            whether self.shots = shot_list
            get y_pred, shot_list
            compute warning_time, true_positive, false_positive

        """

        self.get_y()
        shot_list = self.shots
        y_pred = self.y_pred

        for i in range(len(shot_list)):
            true_disruption = self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.ACTUAL_DISRUPTION_H].tolist()[0]
            if true_disruption == 1 and y_pred[i] == 1:
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.TRUE_POSITIVE_H] = 1
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.FALSE_POSITIVE_H] = 0

            elif true_disruption == 0 and y_pred[i] == 1:
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.TRUE_POSITIVE_H] = 0
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.FALSE_POSITIVE_H] = 1

            elif true_disruption == 0 and y_pred[i] == 0:
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.TRUE_POSITIVE_H] = 0
                self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.FALSE_POSITIVE_H] = 0

    def confusion_matrix(self):
        """
            this function should be called before call self.calc_metrics() ,
            whether self.shots = shot_list
            get y_pred, shot_list
            compute confusion_matrix
        Returns:
            ture postive, false negative, false postive, ture negative

        """

        if len(set(self.get_all_shots(include_all=False)) & set(self.shots)) != len(self.shots):
            self.get_y()
        [[tn, fp], [fn, tp]] = confusion_matrix(self.y_true, self.y_pred)
        return tp, fn, fp, tn

    def ture_positive_rate(self):
        """
            this function should be called before call self.calc_metrics()
            get tp, fn, fp, tn
            compute tpr, fpr

        Returns:
            ture postive rate, false positive rate

        """

        tp, fn, fp, tn = self.confusion_matrix()
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        return tpr, fpr

    def get_accuracy(self):
        """
            this function should be called before call self.calc_metrics()
        Returns:
            accuracy

        """
        if len(set(self.get_all_shots(include_all=False)) & set(self.shots)) != len(self.shots):
            self.get_y()
        accuracy = accuracy_score(y_true=self.y_true, y_pred=self.y_pred, normalize=True,
                                  sample_weight=None)
        return accuracy

    def get_precision(self):
        """
            this function should be called before call self.calc_metrics()
        Returns:
            precision

        """

        if len(set(self.get_all_shots(include_all=False)) & set(self.shots)) != len(self.shots):
            self.get_y()

        precision = precision_score(y_true=self.y_true, y_pred=self.y_pred, average='macro')
        return precision

    def get_recall(self):
        """
            this function should be called before call self.calc_metrics()
        Returns:
            recall

        """
        if len(set(self.get_all_shots(include_all=False)) & set(self.shots)) != len(self.shots):
            self.get_y()

        recall = recall_score(y_true=self.y_true, y_pred=self.y_pred, average='macro')
        return recall

    def warning_time_histogram(self,  time_bins: List[float], file_path = None):
        """
                this function should be called before call self.calc_metrics().
                Plot a column chart, the x-axis is time range,
            the y-axis is the number of shot less in that time period.
        Args:
            file_path:    the path to save .png
            time_bins:    a time endpoint list, unit: s

        """

        matplotlib.use('TkAgg')
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.weight'] = 'bold'
        warning_time_list = []
        shot_list = self.get_all_shots(include_all=False)
        if len(set(self.get_all_shots(include_all=False)) & set(self.shots)) != len(self.shots):
            self.get_y()
        for i in range(len(shot_list)):
            if self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H].tolist()[0] != -1:
                warning_time_list.append(self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H].tolist()[0])
        warning_time_list = np.array(warning_time_list)

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
        if file_path:
            plt.savefig(os.path.join(file_path, 'histogram_warning_time.png'), dpi=300)

    def accumulate_warning_time_plot(self, file_path=None):

        """
                this function should be called before call self.calc_metrics().
                Plot a column chart, the x-axis is time range,
            the y-axis is the number of shot less in that time period.
        Args:
            file_path:    the path to save .png
            true_dis_num:    the number of true_disruptive

        """
        true_dis_num = len(self.get_all_shots(include_all=False))
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.weight'] = 'bold'
        warning_time_list = []
        shot_list = self.get_all_shots(include_all=False)
        if len(set(self.get_all_shots(include_all=False)) & set(self.shots)) != len(self.shots):
            self.get_y()
        for i in range(len(shot_list)):
            if self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H].tolist()[0] != -1:
                warning_time_list.append(self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H].tolist()[0])
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
        if file_path:
            plt.savefig(os.path.join(file_path, 'accumulate_warning_time.png'), dpi=300)



