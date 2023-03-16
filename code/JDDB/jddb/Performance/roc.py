import xlwt
import pandas as pd
import os, sys
from typing import Optional
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib
from .result import Result
from .report import Report
from sklearn.metrics import roc_curve, auc

class Roc:

    def __init__(self, report_csv_path: str):

        self.report_csv_path = report_csv_path

    def roc(self, roc_file_path):
        report = Report(self.report_csv_path).report
        fpr = report.loc['FPR'].tolist()
        tpr = report.loc['TPR'].tolist()
        tpr_ordered = []
        index = np.array(fpr).argsort()
        for i in index:
            tpr_ordered.append(tpr[i])
        fpr.sort()
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(np.array(fpr), np.array(tpr_ordered), color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(roc_file_path, 'Receiver_operating_characteristic.png'), dpi=300)



