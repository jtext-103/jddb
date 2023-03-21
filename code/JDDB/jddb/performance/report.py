from typing import List
import pandas as pd
import os
from result import Result


class Report:

    def __init__(self, report_csv_path: str):

        self.report_csv_path = report_csv_path
        """        
            check if file exist
            if not create df,set threshold nan
            if exists, read it populate self: call read.
        """

        self.__header = ['model_name', 'accuracy', 'precision', 'recall', 'fpr', 'tpr', 'tp', 'fn', 'fp', 'tn']
        self.report = None

    def report_file(self):
        if os.path.exists(self.report_csv_path):
            self.read()
        else:
            self.create_df()

    def create_df(self):
        """
            add header
            set property
        """

        df_report = pd.DataFrame(columns=self.__header)
        self.report = df_report
        self.save()

    def read(self):
        """
            check file format
            read in all module
        """

        self.report = pd.read_excel(self.report_csv_path)
        if self.report.columns is None:
            self.report = pd.DataFrame(columns=self.__header)
        elif len(set(self.report.columns) & set(self.__header)) != len(self.report.columns):
            raise ValueError("The file from csv_path:{} contains unknown information ".format(self.report_csv_path))

    def save(self):
        """
        save report
        """
        self.report.to_excel(self.report_csv_path, index=False)

    def add(self, result, model_name: str, tardy_alarm_threshold, lucky_guess_threshold):
        """
            add new model
        Args:
            result: a csv file that is instantiated
            model_name: a name of str
            tardy_alarm_threshold: unit:s
            lucky_guess_threshold: unit:s

        """
        self.check_repeat(model_name)
        result.tardy_alarm_threshold = tardy_alarm_threshold
        result.lucky_guess_threshold = lucky_guess_threshold
        result.calc_metrics()
        tpr, fpr = result.ture_positive_rate()
        accuracy = result.get_accuracy()
        precision = result.get_precision()
        recall = result.get_recall()
        tp, fn, fp, tn = result.confusion_matrix()
        index = len(self.report)
        self.report.loc[
            index, ['model_name', 'accuracy', 'precision', 'recall', 'fpr', 'tpr', 'tp', 'fn', 'fp',
                    'tn']] = \
            [model_name, accuracy, precision, recall, fpr, tpr, tp, fn, fp, tn]

    def remove(self, model_name: List[str]):
        """
            giving model_name to remove the specified row
        Args:
            model_name: a list of model
        """
        model_name = self.check_unexisted(model_name)
        for i in range(len(model_name)):
            self.report = self.report.drop(self.report[self.report.model_name == model_name[i]].index)

    def check_repeat(self, model_name: str):
        """
            check existing model_name, if existing, raise error
        Args:
            model_name: a list of model
        """
        if model_name in self.report.model_name.tolist():
            raise ValueError("data of model_name:{} has already existed".format(model_name))

    def check_unexisted(self, model_name: List[str]):
        """
            check whehter shot_no repeated
            if repeated, raise error
        Args:
            model_name: a list
        """

        err_list = []
        for i in range(len(model_name)):
            if model_name[i] not in self.report.model_name.tolist():
                err_list.append(model_name[i])
            if len(err_list) > 0:
                raise ValueError("THE data of number or numbers:{} do not exist".format(err_list))


# if __name__ == '__main__':
#     report = Report("G:\datapractice\\test\\report.xlsx")
#     result = Result("G:\datapractice\\test\\test.xlsx")
#     report.report_file()
#     report.add(result, "test1", 0.02, 0.06)
#     report.add(result, "test2", 0.03, 0.2)
#     report.add(result, "test3", 0.02, 0.3)
#     report.add(result, "test4", 0.02, 0.15)
