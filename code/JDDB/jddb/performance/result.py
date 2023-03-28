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
        Assign a str value to the header
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
    TRUE_NEGATIVE_H = 'true_negative'
    FALSE_NEGATIVE_H = 'false_negative'
    TARDY_ALARM_THRESHOLD_H = 'tardy_alarm_threshold'
    LUCKY_GUESS_THRESHOLD_H = 'lucky_guess_threshold'

    def __init__(self, csv_path: str):
        """
            check if file exist
            if not create df,set threshold nan
            if exists, read it populate self: call read
            Set the shot_no of first line to -10 to record the threshold

        Args:
            csv_path: a path to read or create a csv file
        """

        self.tpr = np.nan
        self.fpr = np.nan
        self.accuracy = np.nan
        self.precision = np.nan
        self.recall = np.nan
        self.confusion_matrix = np.nan
        self.average_warning_time = np.nan
        self.median_warning_time = np.nan
        self.tardy_alarm_threshold = np.nan
        self.lucky_guess_threshold = np.nan
        self.csv_path = csv_path
        self.__header = [self.SHOT_NO_H, self.PREDICTED_DISRUPTION_H, self.PREDICTED_DISRUPTION_TIME_H,
                         self.ACTUAL_DISRUPTION_H, self.ACTUAL_DISRUPTION_TIME_H, self.WARNING_TIME_H,
                         self.TRUE_POSITIVE_H, self.FALSE_POSITIVE_H, self.TRUE_NEGATIVE_H,
                         self.FALSE_NEGATIVE_H, self.TARDY_ALARM_THRESHOLD_H, self.LUCKY_GUESS_THRESHOLD_H]
        self.result = None
        self.y_pred = []
        self.y_true = []
        self.ignore_thresholds = False
        if os.path.exists(csv_path):
            self.read()
        else:
            self.result = pd.DataFrame(columns=self.__header)

        new_data = {self.SHOT_NO_H: -10, self.TARDY_ALARM_THRESHOLD_H: self.tardy_alarm_threshold,
                    self.LUCKY_GUESS_THRESHOLD_H: self.lucky_guess_threshold}
        if self.result.empty:
            self.result.loc[0] = new_data
        elif self.result.loc[0, self.SHOT_NO_H] != -10:
            self.result.loc[0] = new_data

    def read(self):
        """
            check file format
            read in      self.tardy_alarm_threshold
                       self.lucky_guess_threshold
            read in all shots
        """

        self.result = pd.read_csv(self.csv_path)
        if self.result.columns is None:
            self.result = pd.DataFrame(columns=self.__header)
        elif len(set(self.result.columns) & set(self.__header)) != len(self.result.columns):
            raise ValueError("The file from csv_path:{} contains unknown information ".format(self.csv_path))
        if not self.result.empty:
            self.tardy_alarm_threshold = self.result.loc[0, self.TARDY_ALARM_THRESHOLD_H]
            self.lucky_guess_threshold = self.result.loc[0, self.LUCKY_GUESS_THRESHOLD_H]

    def save(self):
        """
            the file will be saved in disk after all processing, else not
        """

        self.result.loc[0, [self.TARDY_ALARM_THRESHOLD_H]] = self.lucky_guess_threshold
        self.result.loc[0, [self.LUCKY_GUESS_THRESHOLD_H]] = self.tardy_alarm_threshold
        self.result.to_csv(self.csv_path, index=False)

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
         Args:
            Instantiated meta_db
        """

        shot_list = self.get_all_shots()
        for shot in shot_list:
            true_disruption = meta_db.get_labels(shot)[self.IS_DISRUPT]

            if true_disruption == False:
                true_disruption = 0
                true_downtime = -1
            else:
                true_disruption = 1
                true_downtime = meta_db.get_labels(shot)[self.DOWN_TIME]
            self.result.loc[self.result[self.SHOT_NO_H] == shot, self.ACTUAL_DISRUPTION_H] = true_disruption
            self.result.loc[self.result[self.SHOT_NO_H] == shot, self.ACTUAL_DISRUPTION_TIME_H] = true_downtime

    def get_all_truth_from_file_repo(self, file_repo: FileRepo):
        """
                get all true_disruption and true_downtime of exsit shot number from file_repo, if duplicated overwrite
            Args:
                Instantiated file_repo
        """

        shot_list = file_repo.get_all_shots()
        for shot in shot_list:
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
            shot_list: a list of shot number to remove the corresponding row
        """
        for i in range(len(shot_list)):
            if shot_list[i] in self.result[self.SHOT_NO_H].tolist():
                self.result = self.result.drop(self.result[self.result[self.SHOT_NO_H] == shot_list[i]].index)

    def get_all_shots(self, include_all=True):
        """
                get all shot_list return shot_list
            Args:
                 include_all:
                 if  include_all=True, return all shot of dataframe
                 if  include_all=False, return shot list except that the value of actual_disruption is np.nan
            Returns:
                shot_list: a list of shot number
        """

        shot_list = self.result[self.SHOT_NO_H].tolist()
        shot_list.remove(-10)
        if include_all is False:
            tmp_shot_list = []
            for shot_no in shot_list:
                true_disruption = \
                    self.result.loc[self.result[self.SHOT_NO_H] == shot_no, self.ACTUAL_DISRUPTION_H].tolist()[0]
                if true_disruption == 0 or true_disruption == 1:
                    tmp_shot_list.append(shot_no)
            shot_list = tmp_shot_list
        return shot_list

    def calc_metrics(self):
        """
            this function should be called before setting self.tardy_alarm_threshold and self.lucky_guess_threshold,
            compute warning_time, true_positive, false_positive, true_negative, false_negative
        """

        shot_list = self.get_all_shots(include_all=False)
        for shot in shot_list:
            if self.result.loc[self.result[self.SHOT_NO_H] == shot, self.ACTUAL_DISRUPTION_H].values[0] is not np.nan:
                shot_ans = self.result.loc[self.result[self.SHOT_NO_H] == shot, [self.PREDICTED_DISRUPTION_H,
                                                                                 self.PREDICTED_DISRUPTION_TIME_H,
                                                                                 self.ACTUAL_DISRUPTION_H,
                                                                                 self.ACTUAL_DISRUPTION_TIME_H]].values[
                    0]
                pred, pred_time, truth, truth_downtime = tuple(shot_ans)
                tp, fp, tn, fn, warning_time = self.get_shot_result(pred, pred_time, truth, truth_downtime)
                # if metrics is calculated, save to this row
                self.result.loc[self.result[self.SHOT_NO_H] == shot, [self.TRUE_POSITIVE_H, self.FALSE_POSITIVE_H,
                                                                      self.TRUE_NEGATIVE_H, self.FALSE_NEGATIVE_H,
                                                                      self.WARNING_TIME_H]] = \
                    [tp, fp, tn, fn, warning_time]
                self.y_pred.append(1 * tp + 1 * fp + 0 * tn + 0 * fn)
                self.y_true.append(1 * tp + 0 * fp + 0 * tn + 1 * fn)

        self.confusion_matrix = self.get_confusion_matrix()
        self.tpr, self.fpr = self.get_ture_positive_rate()
        self.accuracy = self.get_accuracy()
        self.precision = self.get_precision()
        self.recall = self.get_recall()
        self.average_warning_time = self.get_average_warning_time()
        self.median_warning_time = self.get_median_warning_time()

    def get_shot_result(self, pred, pred_time, truth, truth_downtime, ignore_threshold=False):

        """
            return a tuple [tp,fp,tn,fn,warning_time]
        """

        warning_time = -1
        tp = np.nan
        fp = -1
        tn = -1
        fn = -1
        lucky_guess = self.lucky_guess_threshold
        tardy_alarm = self.tardy_alarm_threshold
        if ignore_threshold:
            tardy_alarm = 0
            lucky_guess = 1e100
        if truth == 1:
            if pred == 1:
                warning_time = truth_downtime - pred_time
                tp = 0
                fp = 0
                tn = 0
                fn = 1
                if lucky_guess > warning_time > tardy_alarm:
                    fn = 0
                    tp = 1
            else:
                tp = 0
                fp = 0
                tn = 0
                fn = 1
        elif truth == 0:
            if pred == 0:
                tp = 0
                fp = 0
                tn = 1
                fn = 0
            else:
                tp = 0
                fp = 1
                tn = 0
                fn = 0
        return tp, fp, tn, fn, warning_time

    def get_average_warning_time(self):
        """
            compute value of average warning_time
        """

        shot_list = self.result[self.SHOT_NO_H].tolist()
        warning_time_list = []
        for shot in shot_list:
            if self.result.loc[self.result[self.SHOT_NO_H] == shot, self.TRUE_POSITIVE_H].values[0] == 1:
                warning_time_list.append(
                    self.result.loc[self.result[self.SHOT_NO_H] == shot, self.WARNING_TIME_H].values[0])
        return np.mean(warning_time_list)

    def get_median_warning_time(self):
        """
            compute value of average warning_time
        """

        shot_list = self.result[self.SHOT_NO_H].tolist()
        warning_time_list = []
        for shot in shot_list:
            if self.result.loc[self.result[self.SHOT_NO_H] == shot, self.TRUE_POSITIVE_H].values[0] == 1:
                warning_time_list.append(
                    self.result.loc[self.result[self.SHOT_NO_H] == shot, self.WARNING_TIME_H].values[0])
        return np.median(warning_time_list)

    def get_confusion_matrix(self):
        """
            compute confusion_matrix
        Returns:
            ture postive, false negative, false postive, ture negative

        """

        matrix = confusion_matrix(self.y_true, self.y_pred)

        return matrix

    def get_ture_positive_rate(self):
        """
            this function should be called by self.calc_metrics()
            get tp, fn, fp, tn
            compute tpr, fpr

        Returns:
            ture postive rate, false positive rate

        """
        matrix = self.get_confusion_matrix()
        tn = int(matrix[0][0])
        fp = int(matrix[0][1])
        fn = int(matrix[1][0])
        tp = int(matrix[1][1])
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        return tpr, fpr

    def get_accuracy(self):
        """
            this function should be called by self.calc_metrics()
        Returns:
            accuracy

        """
        accuracy = accuracy_score(y_true=self.y_true, y_pred=self.y_pred, normalize=True,
                                  sample_weight=None)
        return accuracy

    def get_precision(self):
        """
            this function should be called by self.calc_metrics()
        Returns:
            precision

        """

        precision = precision_score(y_true=self.y_true, y_pred=self.y_pred, average='macro')
        return precision

    def get_recall(self):
        """
            this function should be called by self.calc_metrics()
        Returns:
            recall

        """

        recall = recall_score(y_true=self.y_true, y_pred=self.y_pred, average='macro')
        return recall

    def plot_warning_time_histogram(self, time_bins: List[float], file_path=None):
        """
                this function should be called before call self.calc_metrics().
                Plot a column chart, the x-axis is time range,
            the y-axis is the number of shot during that waring time threshold.
        Args:
            file_path:    the path to save .png
            time_bins:    a time endpoint list, unit: s

        """

        # matplotlib.use('TkAgg')
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.weight'] = 'bold'
        warning_time_list = []
        shot_list = self.get_all_shots(include_all=False)
        for i in range(len(shot_list)):
            if self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H].tolist()[0] != -1:
                warning_time_list.append(
                    self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H].tolist()[0])
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

    def plot_accumulate_warning_time(self, file_path=None):

        """
                this function should be called before call self.calc_metrics().
                Plot a line chart, the x-axis is time range,
            the y-axis is the percentage of number of shot during that waring time period.
        Args:
            file_path:    the path to save .png

        """

        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.weight'] = 'bold'
        warning_time_list = []
        shot_list = self.get_all_shots(include_all=False)
        for i in range(len(shot_list)):
            if self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H].tolist()[0] != -1:
                warning_time_list.append(
                    self.result.loc[self.result[self.SHOT_NO_H] == shot_list[i], self.WARNING_TIME_H].tolist()[0])

        fn = int(self.confusion_matrix[1][0])
        tp = int(self.confusion_matrix[1][1])
        true_dis_num = fn + tp
        warning_time = np.array(warning_time_list)  # ->s
        warning_time.sort()
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
