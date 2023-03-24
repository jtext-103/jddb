from typing import List
import pandas as pd
import os
from .result import Result
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc

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

    def add(self, result, model_name: str):
        """
            add new model
        Args:
            result: a csv file that is instantiated
            model_name: a name of str
            tardy_alarm_threshold: unit:s
            lucky_guess_threshold: unit:s

        """
        self.check_repeat(model_name)
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

    def roc(self, roc_file_path):
        """
            draw roc curve
        Args:
            roc_file_path: a path to save picture

        """

        fpr = self.report.fpr.tolist()
        tpr = self.report.tpr.tolist()
        tpr_ordered = []
        index = np.array(fpr).argsort()
        for i in index:
            tpr_ordered.append(tpr[i])
        fpr.sort()
        roc_auc = auc(fpr, tpr)
        lw = 2
        #  matplotlib.use('QtAgg')
        plt.rcParams["figure.figsize"] = (10, 10)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(np.array(fpr), np.array(tpr_ordered), color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(roc_file_path, 'Receiver_operating_characteristic.png'), dpi=300)
        plt.show(block=True)


# if __name__ == '__main__':
#     report = Report("G:\datapractice\\test\\report.xlsx")
#     result = Result("G:\datapractice\\test\\test.xlsx")
#     report.report_file()
#     report.add(result, "test1" )
#     report.add(result, "test2")
#     report.add(result, "test3")
#     report.add(result, "test4")
#     report.save()
