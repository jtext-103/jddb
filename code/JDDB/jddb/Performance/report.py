import xlwt
import pandas as pd
import os, sys
from .result import Result
from sklearn.metrics import confusion_matrix
# 调用函数result 返回值修改！！！！！！
class Report:

    def __init__(self, report_csv_path: str):

        self.report_csv_path = report_csv_path
        # check if file exist
        # if not create df,set threshold nan
        # if exists, read it populate self: call read
        self.__header = ['model_name', 'accuracy', 'precision', 'recall', 'fpr', 'tpr', 'tp', 'fn', 'fp', 'tn']
        self.report = None

    def report(self):
        if os.path.exists(self.report_csv_path):
            self.read()
        else:
            self.create_df()

    def create_df(self):
        # add header
        # set property
        df_report = pd.DataFrame(columns=self.__header)
        self.report = df_report
        self.save()

    def read(self):

        # check file format
        # read in all module
        self.report = pd.read_excel(self.report_csv_path)
        if self.report.columns is None:
            self.report = pd.DataFrame(columns=self.__header)
        elif self.report.columns != self.__header:
            raise ValueError("The file from csv_path:{} contains unknown information ".format(self.report_csv_path))


    def save(self):
        self.report.to_excel(self.report_csv_path, index=False)

    def add(self, result_csv_path: str, model_name: Optional[List[str]]):
        self.report = self.report()
        result = Result(result_csv_path)
        self.check_repeat(model_name)

        tpr, fpr = result.ture_positive_rate()
        accuracy = result.get_accuracy()
        precision = result.get_precision()
        recall = result.get_recall()
        tp, fn, fp, tn = result.confusion_matrix()

        self.report.loc[len(report) + 1,  ['model_name', 'accuracy', 'precision', 'recall', 'fpr', 'tpr', 'tp', 'fn', 'fp', 'tn']] = \
            [model_name, accuracy, precision, recall, tpr, fpr, tp, fn, fp, tn]
        self.report.to_excel(self.report_csv_path, index=False)

    def remove(self, model_name: Optional[List[str]]):
        model_name = self.check_no_exist(model_name)
        for i in range(len(model_name)):
            self.report = self.report.drop(self.report[self.report.model_name == model_name[i]].index)

    def check_repeat(self, model_name: Optional[List[str]]):
        err_list = []
        for i in range(len(model_name)):
            if model_name[i] in self.report.model_name.tolist():
                err_list.append(model_name[i])
        if len(err_list) > 0:
            raise ValueError("data of model_name:{} has already existed".format(err_list))

    def check_no_exist(self, model_name: Optional[List[str]] = None):
        # check whehter shot_no repeated
        # if repeated, raise error
        err_list = []
        for i in range(len(model_name)):
            if model_name[i] not in self.report.model_name.tolist():
                err_list.append(model_name[i])
            if len(err_list) > 0:
                raise ValueError("THE data of number or numbers:{} do not exist".format(err_list))



