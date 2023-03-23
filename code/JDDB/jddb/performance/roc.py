import os
import numpy as np
from matplotlib import pyplot as plt
from report import Report
from sklearn.metrics import auc
import matplotlib

class Roc:

    """
        draw roc curve, and save the picture

    """
    def __init__(self, report_csv_path: str):
        """
        Args:
            report_csv_path:  input a report csv path to load data
        """

        self.report_csv_path = report_csv_path

    def roc(self, roc_file_path):
        """
            draw roc curve
        Args:
            roc_file_path: a path to save picture

        """
        report = Report(self.report_csv_path)
        report.report_file()

        fpr = report.report.fpr.tolist()
        tpr = report.report.tpr.tolist()
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
#     report.report_file()
# # #     # report.add("G:\datapractice\\test\\test.xlsx", "test1", 0.02, 0.3)
# # #     # report.add("G:\datapractice\\test\\test.xlsx", "test2", 0.02, 0.3)
# # #     # report.save()
#     roc = Roc("G:\datapractice\\test\\report.xlsx")
#     roc.roc("G:\datapractice\\test")

