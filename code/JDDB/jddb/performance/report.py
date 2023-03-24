from typing import List
import pandas as pd
import os
from .result import Result
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc


class Report:
    MODEL_NAME = 'model_name'
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    FPR = 'fpr'
    TPR = 'tpr'
    TP = 'tp'
    FN = 'fn'
    FP = 'fp'
    TN = 'tn'

    def __init__(self, report_csv_path: str):

        self.report_csv_path = report_csv_path
        """        
            check if file exist
            if not create df,set threshold nan
            if exists, read it populate self: call read.
        """

        self.__header = [self.MODEL_NAME, self.ACCURACY, self.PRECISION, self.RECALL, self.FPR, self.TPR, self.TP,
                         self.FN, self.FP, self.TN]
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
        self.report.to_csv(self.report_csv_path, index=False)

    def add(self, result, model_name: str):
        """
            add new model
        Args:
            result: a csv file that is instantiated
            model_name: a name of str

        """

        result.calc_metrics()
        tpr, fpr = result.ture_positive_rate()
        accuracy = result.get_accuracy()
        precision = result.get_precision()
        recall = result.get_recall()
        tp, fn, fp, tn = result.confusion_matrix()
        index = len(self.report)
        # check if key already exists in the dataframe
        if model_name in self.report[self.MODEL_NAME].values:
            # update the row with matching key
            self.report.loc[self.report[self.MODEL_NAME] == model_name, [self.MODEL_NAME, self.ACCURACY, self.PRECISION,
                                                                         self.RECALL,
                                                                         self.FPR, self.TPR, self.TP, self.FN, self.FP,
                                                                         self.TN]] = \
                [model_name, accuracy, precision, recall, fpr, tpr, tp, fn, fp, tn]
        else:
            # insert new row
            new_row = {self.MODEL_NAME: model_name, self.ACCURACY: accuracy, self.PRECISION: precision,
                       self.RECALL: recall,
                       self.FPR: fpr, self.TPR: tpr, self.TP: tp, self.FN: fn, self.FP: fp, self.TN: tn}
            self.report = self.report.append(new_row, ignore_index=True)

    def remove(self, model_name: List[str]):
        """
            giving model_name to remove the specified row
        Args:
            model_name: a list of model
        """
        if model_name in self.report[self.MODEL_NAME].values:
            for i in range(len(model_name)):
                self.report = self.report.drop(self.report[self.report[self.MODEL_NAME] == model_name[i]].index)
        else:
            pass

    def roc(self, roc_file_path =None):
        """
            draw roc curve
        Args:
            roc_file_path: a path to save picture

        """

        fpr = self.report[self.FPR].tolist()
        tpr = self.report[self.TPR].tolist()
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
        if roc_file_path:
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
